#pragma once

#include "mecan.h"
#include <string>
#include <cstdint>

#if defined(_WIN32)
#define MECAN_EXPORT __declspec(dllexport)
#else
#define MECAN_EXPORT
#endif

// This is a hand-coded light-weight Python C-API bridge.
// It allows TSSR scripts to call the TST C++ kernels directly.

extern "C" {

    // Tensor Creation
    MECAN_EXPORT void* mt_create_tensor(int64_t* shape, int ndim, int dtype, int device);
    MECAN_EXPORT void mt_destroy_tensor(void* tensor_ptr);
    
    // Memory Access
    MECAN_EXPORT float* mt_get_data_ptr_f32(void* tensor_ptr);
    
    // Math Operations
    MECAN_EXPORT void mt_op_add(void* a, void* b, void* out);
    MECAN_EXPORT void mt_op_matmul(void* a, void* b, void* out);

    // Metadata
    MECAN_EXPORT int64_t mt_get_numel(void* tensor_ptr);

    // Native .mt binary weight codec
    MECAN_EXPORT void mt_save_paged_mt(void* tensor_ptr, const char* filepath, const char* tensor_name, int quant_scheme, uint64_t page_bytes);
    MECAN_EXPORT void* mt_load_paged_mt(const char* filepath);

}
