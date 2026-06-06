#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "mecan/runtime/interfaces.h"

namespace mecan {
namespace runtime {

class BudgetController {
public:
    explicit BudgetController(MemoryBudget budget);

    bool can_reserve(MemoryTier tier, uint64_t bytes) const;
    bool reserve(MemoryTier tier, uint64_t bytes);
    void release(MemoryTier tier, uint64_t bytes);
    uint64_t used_bytes(MemoryTier tier) const;
    const MemoryBudget& config() const { return budget_; }

private:
    MemoryBudget budget_;
    uint64_t used_vram_ = 0;
    uint64_t used_ram_ = 0;
    uint64_t used_nvme_ = 0;
};

class TieredMemoryPager : public IMemoryPager {
public:
    explicit TieredMemoryPager(MemoryBudget budget);

    void register_tensor(const PagedTensorMetadata& metadata) override;
    void ensure_resident(const std::string& tensor_name, MemoryTier target) override;
    void prefetch(const std::string& tensor_name, size_t page_begin, size_t page_count, MemoryTier target) override;
    void evict(const std::string& tensor_name, MemoryTier target) override;
    uint64_t used_bytes(MemoryTier tier) const override;
    MemoryBudget budget() const override;

private:
    struct TensorState {
        PagedTensorMetadata metadata;
        std::vector<MemoryTier> page_tiers;
    };

    void move_page(TensorState& state, size_t page_idx, MemoryTier target);
    static uint64_t page_bytes(const TensorState& state, size_t page_idx);

    mutable std::mutex mu_;
    BudgetController budget_;
    std::unordered_map<std::string, TensorState> tensors_;
};

} // namespace runtime
} // namespace mecan
