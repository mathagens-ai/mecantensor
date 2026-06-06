"""
MecanTensor QSBits — 1-bit XNOR+POPCOUNT Binary Engine (Python Wrapper)

Wraps the C++ QSBits kernel (qsbits_forward).
Executes neural network forward passes using binary logic:
  output[b][o] = POPCOUNT(XNOR(input[b], weights[o]))

Usage:
    from mecantensor import qsbits
    output = qsbits.forward(input_packed, weights_packed)
"""

import numpy as np
from ._native import get_lib
import ctypes


def pack_binary(x):
    """
    Pack a float/int tensor into binary uint8 format.
    Values > 0 become 1, values <= 0 become 0.
    Packs 8 values per byte.
    """
    x = np.asarray(x)
    binary = (x > 0).astype(np.uint8).flatten()
    # Pad to multiple of 8
    pad = (8 - len(binary) % 8) % 8
    if pad:
        binary = np.concatenate([binary, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(binary)


def forward(input_packed, weights_packed, output_dim=None):
    """
    QSBits forward pass: XNOR + POPCOUNT.

    Args:
        input_packed: numpy uint8 array (batch, K_packed) — packed binary input
        weights_packed: numpy uint8 array (O, K_packed) — packed binary weights
        output_dim: number of output neurons (inferred from weights if None)

    Returns:
        output: numpy float32 (batch, O)
    """
    if input_packed.ndim == 1:
        input_packed = input_packed.reshape(1, -1)

    O = weights_packed.shape[0] if output_dim is None else output_dim
    K_packed = weights_packed.shape[1]
    batch = input_packed.shape[0]

    # Try C++ backend
    lib = get_lib()
    if lib is not None and hasattr(lib, 'mt_op_qsbits'):
        output = np.zeros((batch, O), dtype=np.float32)

        a_shape = (ctypes.c_int64 * 2)(batch, K_packed)
        w_shape = (ctypes.c_int64 * 2)(O, K_packed)
        o_shape = (ctypes.c_int64 * 2)(batch, O)

        lib.mt_create_tensor.restype = ctypes.c_void_p
        lib.mt_get_data_ptr_u8.restype = ctypes.c_void_p
        lib.mt_get_data_ptr_u8.argtypes = [ctypes.c_void_p]
        lib.mt_get_data_ptr_f32.restype = ctypes.c_void_p
        lib.mt_get_data_ptr_f32.argtypes = [ctypes.c_void_p]
        lib.mt_op_qsbits.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        lib.mt_destroy_tensor.argtypes = [ctypes.c_void_p]
        
        # dtype=5 = Int8 for packed binary, 1 = Float32
        a_ptr = lib.mt_create_tensor(a_shape, 2, 5, 0)
        w_ptr = lib.mt_create_tensor(w_shape, 2, 5, 0)
        o_ptr = lib.mt_create_tensor(o_shape, 2, 1, 0)  # output in FP32

        try:
            # Copy packed data (raw bytes)
            a_raw = lib.mt_get_data_ptr_u8(a_ptr)
            w_raw = lib.mt_get_data_ptr_u8(w_ptr)
            ctypes.memmove(a_raw, input_packed.ctypes.data, input_packed.nbytes)
            ctypes.memmove(w_raw, weights_packed.ctypes.data, weights_packed.nbytes)

            lib.mt_op_qsbits(a_ptr, w_ptr, o_ptr)

            o_data = lib.mt_get_data_ptr_f32(o_ptr)
            ctypes.memmove(output.ctypes.data, o_data, output.nbytes)
        finally:
            lib.mt_destroy_tensor(a_ptr)
            lib.mt_destroy_tensor(w_ptr)
            lib.mt_destroy_tensor(o_ptr)

        return output

    # Python fallback: XNOR + POPCOUNT in numpy
    output = np.zeros((batch, O), dtype=np.float32)
    for b in range(batch):
        for o in range(O):
            xnor_result = ~(input_packed[b] ^ weights_packed[o])
            popcount = sum(bin(byte).count('1') for byte in xnor_result.astype(np.uint8))
            output[b, o] = float(popcount)

    return output


def info():
    """Report QSBits engine status."""
    lib = get_lib()
    native = lib is not None and hasattr(lib, 'mt_op_qsbits')
    scaled = lib is not None and hasattr(lib, 'mt_op_qsbits_scaled')
    print(f"QSBits Engine: {'Active' if native else 'Fallback (Python)'}")
    print(f"QSBits Scaled: {'Active' if scaled else 'Fallback (Python)'}")


def forward_scaled(input_packed, weights_packed, scales, input_scale=1.0,
                    group_size=None, output_dim=None):
    """
    Block-scaled QSBits forward pass.

    Divides the packed dimension into groups, computes XNOR+POPCOUNT per group,
    and scales each group by a learned FP32 factor.

    Args:
        input_packed: numpy uint8 (batch, K_packed)
        weights_packed: numpy uint8 (O, K_packed)
        scales: numpy float32 (O, N_groups)
        input_scale: float, global activation scale factor
        group_size: int, number of binary elements per group (must be multiple of 8)
        output_dim: int, inferred from weights if None

    Returns:
        output: numpy float32 (batch, O)
    """
    if input_packed.ndim == 1:
        input_packed = input_packed.reshape(1, -1)

    O = weights_packed.shape[0] if output_dim is None else output_dim
    K_packed = weights_packed.shape[1]
    batch = input_packed.shape[0]

    if group_size is None:
        group_size = K_packed * 8

    g_packed = group_size // 8
    n_groups = K_packed // g_packed

    lib = get_lib()
    if lib is not None and hasattr(lib, 'mt_op_qsbits_scaled'):
        output = np.zeros((batch, O), dtype=np.float32)
        scales = np.ascontiguousarray(scales, dtype=np.float32)

        a_shape = (ctypes.c_int64 * 2)(batch, K_packed)
        w_shape = (ctypes.c_int64 * 2)(O, K_packed)
        s_shape = (ctypes.c_int64 * 2)(O, n_groups)
        o_shape = (ctypes.c_int64 * 2)(batch, O)

        lib.mt_create_tensor.restype = ctypes.c_void_p
        lib.mt_get_data_ptr_u8.restype = ctypes.c_void_p
        lib.mt_get_data_ptr_u8.argtypes = [ctypes.c_void_p]
        lib.mt_get_data_ptr_f32.restype = ctypes.c_void_p
        lib.mt_get_data_ptr_f32.argtypes = [ctypes.c_void_p]
        lib.mt_op_qsbits_scaled.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_float, ctypes.c_int, ctypes.c_void_p
        ]
        lib.mt_destroy_tensor.argtypes = [ctypes.c_void_p]

        # dtype 5 = Int8/packed binary, 1 = Float32
        a_ptr = lib.mt_create_tensor(a_shape, 2, 5, 0)
        w_ptr = lib.mt_create_tensor(w_shape, 2, 5, 0)
        s_ptr = lib.mt_create_tensor(s_shape, 2, 1, 0)
        o_ptr = lib.mt_create_tensor(o_shape, 2, 1, 0)

        try:
            a_raw = lib.mt_get_data_ptr_u8(a_ptr)
            w_raw = lib.mt_get_data_ptr_u8(w_ptr)
            s_raw = lib.mt_get_data_ptr_f32(s_ptr)
            ctypes.memmove(a_raw, input_packed.ctypes.data, input_packed.nbytes)
            ctypes.memmove(w_raw, weights_packed.ctypes.data, weights_packed.nbytes)
            ctypes.memmove(s_raw, scales.ctypes.data, scales.nbytes)

            lib.mt_op_qsbits_scaled(
                a_ptr, w_ptr, s_ptr,
                ctypes.c_float(input_scale),
                ctypes.c_int(group_size),
                o_ptr
            )

            o_data = lib.mt_get_data_ptr_f32(o_ptr)
            ctypes.memmove(output.ctypes.data, o_data, output.nbytes)
        finally:
            lib.mt_destroy_tensor(a_ptr)
            lib.mt_destroy_tensor(w_ptr)
            lib.mt_destroy_tensor(s_ptr)
            lib.mt_destroy_tensor(o_ptr)

        return output

    # NumPy fallback
    output = np.zeros((batch, O), dtype=np.float32)
    for b in range(batch):
        for o in range(O):
            acc = 0.0
            for g in range(n_groups):
                base = g * g_packed
                xnor_chunk = ~(input_packed[b, base:base + g_packed] ^ weights_packed[o, base:base + g_packed])
                pop = sum(bin(byte).count('1') for byte in xnor_chunk.astype(np.uint8))
                val = 2.0 * pop - group_size
                acc += val * scales[o, g]
            output[b, o] = acc * input_scale

    return output
