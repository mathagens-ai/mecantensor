#include "autograd/engine.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace mecan {
namespace autograd {

namespace {

void dfs_topo(const std::shared_ptr<Function>& fn,
              std::unordered_set<Function*>& visited,
              std::vector<std::shared_ptr<Function>>& topo) {
    if (!fn || visited.count(fn.get()) > 0) {
        return;
    }
    visited.insert(fn.get());

    for (const auto& next : fn->next_functions) {
        dfs_topo(next, visited, topo);
    }
    topo.push_back(fn);
}

} // namespace

Tensor Engine::make_seed_grad(const Tensor& reference) {
    Tensor grad(reference.shape(), reference.dtype(), reference.device());

    if (reference.dtype() == core::ScalarType::Float32) {
        float* grad_ptr = grad.data_ptr<float>();
        for (size_t i = 0; i < grad.numel(); ++i) {
            grad_ptr[i] = 1.0f;
        }
        return grad;
    }

    throw std::runtime_error("Autograd seed currently supports Float32 tensors only.");
}

void Engine::add_inplace(Tensor& dst, const Tensor& src) {
    if (dst.numel() != src.numel()) {
        throw std::runtime_error("Autograd gradient accumulation shape mismatch.");
    }
    if (dst.dtype() != src.dtype()) {
        throw std::runtime_error("Autograd gradient accumulation dtype mismatch.");
    }

    if (dst.dtype() == core::ScalarType::Float32) {
        float* dst_ptr = dst.data_ptr<float>();
        const float* src_ptr = src.data_ptr<float>();
        for (size_t i = 0; i < dst.numel(); ++i) {
            dst_ptr[i] += src_ptr[i];
        }
        return;
    }

    throw std::runtime_error("Autograd accumulation currently supports Float32 tensors only.");
}

std::vector<std::shared_ptr<Function>> Engine::topological_sort(const std::shared_ptr<Function>& root) {
    std::vector<std::shared_ptr<Function>> topo;
    std::unordered_set<Function*> visited;
    dfs_topo(root, visited, topo);
    return topo;
}

void Engine::backward(Variable& v) {
    if (!v.requires_grad) {
        return;
    }

    if (!v.grad_fn) {
        // Leaf scalar/tensor backward support.
        v.grad = make_seed_grad(v.data);
        return;
    }

    const Tensor seed = make_seed_grad(v.data);
    std::unordered_map<Function*, Variable> accumulated_grads;
    accumulated_grads.emplace(v.grad_fn.get(), Variable(seed, false));

    const std::vector<std::shared_ptr<Function>> topo = topological_sort(v.grad_fn);
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        const auto& fn = *it;
        auto grad_it = accumulated_grads.find(fn.get());
        if (grad_it == accumulated_grads.end()) {
            continue;
        }

        std::vector<Variable> propagated = fn->apply({grad_it->second});
        const size_t edge_count = std::min(propagated.size(), fn->next_functions.size());

        for (size_t i = 0; i < edge_count; ++i) {
            const auto& next_fn = fn->next_functions[i];
            if (!next_fn) {
                continue;
            }

            auto next_it = accumulated_grads.find(next_fn.get());
            if (next_it == accumulated_grads.end()) {
                accumulated_grads.emplace(next_fn.get(), propagated[i]);
            } else {
                add_inplace(next_it->second.data, propagated[i].data);
            }
        }
    }
}

} // namespace autograd
} // namespace mecan
