#pragma once

#include <atomic>
#include "allocator.h"

namespace mecan {
namespace core {

    // Storage holds the actual raw memory buffer.
    // It is reference-counted so multiple Tensors can safely view the same memory.
    class Storage {
    private:
        void* data_;
        size_t nbytes_;
        Allocator* allocator_;
        Device device_;
        ScalarType dtype_;

    public:
        Storage(size_t nbytes, Allocator* allocator, ScalarType dtype)
            : nbytes_(nbytes), allocator_(allocator), device_(allocator->device()), dtype_(dtype) {
            data_ = allocator_->allocate(nbytes);
        }

        ~Storage() {
            if (data_ && allocator_) {
                allocator_->deallocate(data_);
                data_ = nullptr;
            }
        }

        // Disable copy
        Storage(const Storage&) = delete;
        Storage& operator=(const Storage&) = delete;

        inline void* data() const { return data_; }
        inline size_t nbytes() const { return nbytes_; }
        inline Device device() const { return device_; }
        inline ScalarType dtype() const { return dtype_; }
    };

} // namespace core
} // namespace mecan
