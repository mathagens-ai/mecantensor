#include "mecan/runtime/pager.h"

#include <algorithm>
#include <stdexcept>

namespace mecan {
namespace runtime {

namespace {
uint64_t& used_for_tier(MemoryTier tier, uint64_t& vram, uint64_t& ram, uint64_t& nvme) {
    switch (tier) {
        case MemoryTier::VRAM: return vram;
        case MemoryTier::RAM: return ram;
        case MemoryTier::NVME: return nvme;
        default: return ram;
    }
}

uint64_t budget_for_tier(MemoryTier tier, const MemoryBudget& budget) {
    switch (tier) {
        case MemoryTier::VRAM: return budget.vram_bytes;
        case MemoryTier::RAM: return budget.ram_bytes;
        case MemoryTier::NVME: return budget.nvme_bytes;
        default: return budget.ram_bytes;
    }
}
} // namespace

BudgetController::BudgetController(MemoryBudget budget)
    : budget_(budget) {}

bool BudgetController::can_reserve(MemoryTier tier, uint64_t bytes) const {
    const uint64_t limit = budget_for_tier(tier, budget_);
    const uint64_t used = used_bytes(tier);
    if (limit == 0) {
        return !budget_.hard_limit;
    }
    return (used + bytes) <= limit;
}

bool BudgetController::reserve(MemoryTier tier, uint64_t bytes) {
    if (!can_reserve(tier, bytes)) {
        return false;
    }
    used_for_tier(tier, used_vram_, used_ram_, used_nvme_) += bytes;
    return true;
}

void BudgetController::release(MemoryTier tier, uint64_t bytes) {
    uint64_t& used = used_for_tier(tier, used_vram_, used_ram_, used_nvme_);
    used = (bytes > used) ? 0 : (used - bytes);
}

uint64_t BudgetController::used_bytes(MemoryTier tier) const {
    switch (tier) {
        case MemoryTier::VRAM: return used_vram_;
        case MemoryTier::RAM: return used_ram_;
        case MemoryTier::NVME: return used_nvme_;
        default: return used_ram_;
    }
}

TieredMemoryPager::TieredMemoryPager(MemoryBudget budget)
    : budget_(budget) {}

void TieredMemoryPager::register_tensor(const PagedTensorMetadata& metadata) {
    if (metadata.tensor_name.empty()) {
        throw std::runtime_error("TieredMemoryPager: tensor_name cannot be empty.");
    }
    if (metadata.page_layout.page_bytes == 0) {
        throw std::runtime_error("TieredMemoryPager: page size cannot be zero.");
    }

    std::lock_guard<std::mutex> lock(mu_);
    TensorState state;
    state.metadata = metadata;

    const uint64_t nbytes = metadata.total_nbytes;
    const uint64_t page_sz = metadata.page_layout.page_bytes;
    const size_t num_pages = static_cast<size_t>((nbytes + page_sz - 1) / page_sz);
    state.page_tiers.assign(num_pages, MemoryTier::NVME);

    if (!budget_.reserve(MemoryTier::NVME, nbytes)) {
        throw std::runtime_error("TieredMemoryPager: NVMe budget exceeded while registering tensor.");
    }
    tensors_[metadata.tensor_name] = std::move(state);
}

void TieredMemoryPager::ensure_resident(const std::string& tensor_name, MemoryTier target) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = tensors_.find(tensor_name);
    if (it == tensors_.end()) {
        throw std::runtime_error("TieredMemoryPager: tensor not found: " + tensor_name);
    }
    for (size_t i = 0; i < it->second.page_tiers.size(); ++i) {
        move_page(it->second, i, target);
    }
}

void TieredMemoryPager::prefetch(const std::string& tensor_name, size_t page_begin, size_t page_count, MemoryTier target) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = tensors_.find(tensor_name);
    if (it == tensors_.end()) {
        throw std::runtime_error("TieredMemoryPager: tensor not found: " + tensor_name);
    }
    const size_t end = std::min(it->second.page_tiers.size(), page_begin + page_count);
    for (size_t i = page_begin; i < end; ++i) {
        move_page(it->second, i, target);
    }
}

void TieredMemoryPager::evict(const std::string& tensor_name, MemoryTier target) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = tensors_.find(tensor_name);
    if (it == tensors_.end()) {
        throw std::runtime_error("TieredMemoryPager: tensor not found: " + tensor_name);
    }
    for (size_t i = 0; i < it->second.page_tiers.size(); ++i) {
        move_page(it->second, i, target);
    }
}

uint64_t TieredMemoryPager::used_bytes(MemoryTier tier) const {
    std::lock_guard<std::mutex> lock(mu_);
    return budget_.used_bytes(tier);
}

MemoryBudget TieredMemoryPager::budget() const {
    std::lock_guard<std::mutex> lock(mu_);
    return budget_.config();
}

void TieredMemoryPager::move_page(TensorState& state, size_t page_idx, MemoryTier target) {
    const MemoryTier current = state.page_tiers[page_idx];
    if (current == target) {
        return;
    }

    const uint64_t bytes = page_bytes(state, page_idx);
    if (!budget_.reserve(target, bytes)) {
        if (budget_.config().hard_limit) {
            throw std::runtime_error(
                "TieredMemoryPager: memory budget exceeded while moving page to target tier.");
        }
        return;
    }

    budget_.release(current, bytes);
    state.page_tiers[page_idx] = target;
}

uint64_t TieredMemoryPager::page_bytes(const TensorState& state, size_t page_idx) {
    const uint64_t total = state.metadata.total_nbytes;
    const uint64_t page_sz = state.metadata.page_layout.page_bytes;
    const uint64_t offset = static_cast<uint64_t>(page_idx) * page_sz;
    if (offset >= total) {
        return 0;
    }
    return std::min(page_sz, total - offset);
}

} // namespace runtime
} // namespace mecan
