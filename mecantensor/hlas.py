"""
MecanTensor HLAS — Hardware Linear Algebra Subroutines (Python Wrapper)

Wraps the C++ HLAS engine (cpu_engine, gpu_engine, hybrid_engine, npu_engine)
via ctypes. Provides sgemm, conv2d, and quantized matmul from Python.

If the native library is not compiled, falls back to numpy.

Usage:
    from mecantensor import hlas
    C = hlas.sgemm(A, B)
"""

import numpy as np
from ._native import get_lib
import ctypes


def sgemm(A, B, alpha=1.0, beta=0.0):
    """
    Single-precision General Matrix Multiply.
    Routes through the C++ HLAS engine if available, else numpy.

    Args:
        A: numpy float32 array (M, K)
        B: numpy float32 array (K, N)

    Returns:
        C: numpy float32 array (M, N)
    """
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)

    lib = get_lib()
    if lib is not None and hasattr(lib, 'mt_op_matmul'):
        # Use the C++ bridge for matmul
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, f"Shape mismatch: A({M},{K}) @ B({K2},{N})"
        C = np.zeros((M, N), dtype=np.float32)

        # Create temporary tensors via the bridge
        a_shape = (ctypes.c_int64 * 2)(M, K)
        b_shape = (ctypes.c_int64 * 2)(K, N)
        c_shape = (ctypes.c_int64 * 2)(M, N)

        # dtype=1 (Float32), device=0 (CPU)
        a_ptr = lib.mt_create_tensor(a_shape, 2, 1, 0)
        b_ptr = lib.mt_create_tensor(b_shape, 2, 1, 0)
        c_ptr = lib.mt_create_tensor(c_shape, 2, 1, 0)

        try:
            # Copy data in
            lib.mt_get_data_ptr_f32.restype = ctypes.c_void_p
            lib.mt_get_data_ptr_f32.argtypes = [ctypes.c_void_p]
            lib.mt_create_tensor.restype = ctypes.c_void_p
            
            a_data = lib.mt_get_data_ptr_f32(a_ptr)
            b_data = lib.mt_get_data_ptr_f32(b_ptr)
            ctypes.memmove(a_data, A.ctypes.data, A.nbytes)
            ctypes.memmove(b_data, B.ctypes.data, B.nbytes)

            # Execute matmul
            lib.mt_op_matmul(a_ptr, b_ptr, c_ptr)

            # Copy result out
            c_data = lib.mt_get_data_ptr_f32(c_ptr)
            ctypes.memmove(C.ctypes.data, c_data, C.nbytes)
        finally:
            lib.mt_destroy_tensor(a_ptr)
            lib.mt_destroy_tensor(b_ptr)
            lib.mt_destroy_tensor(c_ptr)

        return C

    # Numpy fallback
    return (alpha * (A @ B) + beta).astype(np.float32)


def add(A, B):
    """Element-wise addition via HLAS engine or numpy."""
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)

    lib = get_lib()
    if lib is not None and hasattr(lib, 'mt_op_add'):
        C = np.zeros_like(A)
        a_shape = (ctypes.c_int64 * len(A.shape))(*A.shape)
        b_shape = (ctypes.c_int64 * len(B.shape))(*B.shape)
        c_shape = (ctypes.c_int64 * len(C.shape))(*C.shape)

        # dtype=1 is Float32
        a_ptr = lib.mt_create_tensor(a_shape, len(A.shape), 1, 0)
        b_ptr = lib.mt_create_tensor(b_shape, len(B.shape), 1, 0)
        c_ptr = lib.mt_create_tensor(c_shape, len(C.shape), 1, 0)

        try:
            lib.mt_get_data_ptr_f32.restype = ctypes.c_void_p
            lib.mt_get_data_ptr_f32.argtypes = [ctypes.c_void_p]
            
            ctypes.memmove(lib.mt_get_data_ptr_f32(a_ptr), A.ctypes.data, A.nbytes)
            ctypes.memmove(lib.mt_get_data_ptr_f32(b_ptr), B.ctypes.data, B.nbytes)
            lib.mt_op_add(a_ptr, b_ptr, c_ptr)
            ctypes.memmove(C.ctypes.data, lib.mt_get_data_ptr_f32(c_ptr), C.nbytes)
        finally:
            lib.mt_destroy_tensor(a_ptr)
            lib.mt_destroy_tensor(b_ptr)
            lib.mt_destroy_tensor(c_ptr)
        return C

    return (A + B).astype(np.float32)


def is_native():
    """Check if HLAS is running on the C++ backend."""
    lib = get_lib()
    return lib is not None and hasattr(lib, 'mt_op_matmul')
