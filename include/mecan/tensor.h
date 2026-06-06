#pragma once

#include <vector>
#include <memory>
#include <numeric>
#include "core/storage.h"
#include "autograd/node.h"

namespace mecan {

    class Tensor {
    private:
        std::shared_ptr<core::Storage> storage_;
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
        size_t storage_offset_;

        // Autograd internal state
        bool requires_grad_ = false;
        std::shared_ptr<autograd::Node> grad_fn_ = nullptr;
        std::shared_ptr<Tensor> grad_ = nullptr;

        void compute_strides() {
            strides_.resize(shape_.size());
            size_t stride = 1;
            for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
                strides_[i] = stride;
                stride *= shape_[i];
            }
        }

    public:
        // Create an empty, uninitialized tensor
        Tensor() : storage_offset_(0) {}

        // Create a new tensor with allocated memory
        Tensor(std::vector<size_t> shape, core::ScalarType dtype, core::Device device = core::Device(core::DeviceType::CPU));

        // Create a tensor VIEW (zero-copy)
        Tensor(std::shared_ptr<core::Storage> storage, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset);

        // Core properties
        inline const std::vector<size_t>& shape() const { return shape_; }
        inline const std::vector<size_t>& strides() const { return strides_; }
        inline size_t ndimension() const { return shape_.size(); }
        inline size_t size(size_t dim) const { return shape_.at(dim); }
        inline size_t stride(size_t dim) const { return strides_.at(dim); }
        inline size_t storage_offset() const { return storage_offset_; }
        inline bool defined() const { return static_cast<bool>(storage_); }
        inline core::ScalarType dtype() const { return storage_->dtype(); }
        inline core::Device device() const { return storage_->device(); }

        // Total elements
        inline size_t numel() const {
            if (shape_.empty()) return 0;
            return std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<size_t>());
        }

        // Pointer access
        template <typename T>
        inline T* data_ptr() {
            return reinterpret_cast<T*>(static_cast<char*>(storage_->data()) + storage_offset_ * sizeof(T));
        }

        template <typename T>
        inline const T* data_ptr() const {
            return reinterpret_cast<const T*>(static_cast<const char*>(storage_->data()) + storage_offset_ * sizeof(T));
        }

        // Fast view modifications (No physical memory movement)
        Tensor view(std::vector<size_t> new_shape) const;
        Tensor reshape(std::vector<size_t> new_shape) const;
        void resize_(std::vector<size_t> new_shape);
        Tensor transpose(size_t dim0, size_t dim1) const;

        // ─── Autograd Interface ───────────────────────────────────────────────
        inline bool requires_grad() const { return requires_grad_; }
        void set_requires_grad(bool req) { requires_grad_ = req; }

        inline std::shared_ptr<autograd::Node> grad_fn() const { return grad_fn_; }
        void set_grad_fn(std::shared_ptr<autograd::Node> fn) { grad_fn_ = std::move(fn); }

        inline Tensor grad() const { return grad_ ? *grad_ : Tensor(); }
        void set_grad(const Tensor& g) { 
            if (!grad_) grad_ = std::make_shared<Tensor>();
            *grad_ = g; 
        }

        // Execute the backward pass (Autograd engine entry point)
        void backward();
    };

} // namespace mecan
