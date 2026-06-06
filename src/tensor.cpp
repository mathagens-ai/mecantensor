#include "tensor.h"
#include <stdexcept>
#include <iostream>
#include <utility>

namespace mecan {

    Tensor::Tensor(std::vector<size_t> shape, core::ScalarType dtype, core::Device device)
        : shape_(shape), storage_offset_(0) {
        
        compute_strides();
        size_t total_elements = numel();
        size_t nbytes = total_elements * core::element_size(dtype);
        
        core::Allocator* allocator = core::get_allocator(device.type);
        storage_ = std::make_shared<core::Storage>(nbytes, allocator, dtype);
    }

    Tensor::Tensor(std::shared_ptr<core::Storage> storage, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset)
        : storage_(storage), shape_(shape), strides_(strides), storage_offset_(offset) {}

    Tensor Tensor::view(std::vector<size_t> new_shape) const {
        size_t new_numel = 1;
        for (auto s : new_shape) new_numel *= s;
        
        if (new_numel != numel()) {
            throw std::runtime_error("TST Error: View size mismatch.");
        }
        
        Tensor v(storage_, new_shape, {}, storage_offset_);
        v.compute_strides();
        return v;
    }

    Tensor Tensor::reshape(std::vector<size_t> new_shape) const {
        return view(std::move(new_shape));
    }

    void Tensor::resize_(std::vector<size_t> new_shape) {
        size_t new_numel = 1;
        for (auto s : new_shape) new_numel *= s;

        if (new_numel != numel()) {
            throw std::runtime_error("TST Error: resize_ only supports metadata-preserving reshapes.");
        }
        shape_ = std::move(new_shape);
        compute_strides();
    }

    Tensor Tensor::transpose(size_t dim0, size_t dim1) const {
        if (dim0 >= shape_.size() || dim1 >= shape_.size()) {
            throw std::out_of_range("TST Error: Transpose dimensions out of bounds.");
        }
        
        std::vector<size_t> new_shape = shape_;
        std::vector<size_t> new_strides = strides_;
        
        std::swap(new_shape[dim0], new_shape[dim1]);
        std::swap(new_strides[dim0], new_strides[dim1]);
        
        return Tensor(storage_, new_shape, new_strides, storage_offset_);
    }

} // namespace mecan
