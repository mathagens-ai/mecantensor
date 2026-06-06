"""
MecanTensor IO — .mt Paged Serialization (Python Wrapper)

Wraps the C++ save/load functions for the .mt (MecanTensor) format.
Supports both the simple .mt format and the paged .mt format
for streaming 100B+ parameter models from SSD.

Usage:
    from mecantensor import io
    io.save(tensor_np, "model.mt")
    loaded = io.load("model.mt")
"""

import numpy as np
from ._native import get_lib
import ctypes
import struct
import os


# .mt magic header
MT_MAGIC = b"MECAN_MT1"
MT_VERSION = 1

# DType mapping (matches core::ScalarType in C++)
_DTYPE_MAP = {
    0: np.float32,
    1: np.float64,
    2: np.float16,
    3: np.int32,
    4: np.int8,
    5: np.uint8,
}

_DTYPE_REVERSE = {v: k for k, v in _DTYPE_MAP.items()}


def save(tensor, filepath, name="tensor"):
    """
    Save a numpy tensor in .mt format.
    Tries the C++ backend first; falls back to pure Python.

    Args:
        tensor: numpy array
        filepath: path to save (should end in .mt)
        name: optional tensor name for metadata
    """
    if not filepath.endswith('.mt'):
        filepath += '.mt'

    lib = get_lib()
    if lib is not None and hasattr(lib, 'mt_save_paged_mt'):
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        shape = tensor.shape
        a_shape = (ctypes.c_int64 * len(shape))(*shape)
        ptr = lib.mt_create_tensor(a_shape, len(shape), 0, 0)
        try:
            lib.mt_get_data_ptr_f32.restype = ctypes.POINTER(ctypes.c_float)
            ctypes.memmove(lib.mt_get_data_ptr_f32(ptr), tensor.ctypes.data, tensor.nbytes)
            lib.mt_save_paged_mt(
                ptr,
                filepath.encode('utf-8'),
                name.encode('utf-8'),
                0,  # quant_scheme = NONE
                1 << 20  # page_bytes = 1MB
            )
        finally:
            lib.mt_destroy_tensor(ptr)
        return

    # Python fallback: write .mt binary format
    tensor = np.ascontiguousarray(tensor)
    dtype_code = _DTYPE_REVERSE.get(tensor.dtype.type, 0)

    with open(filepath, 'wb') as f:
        f.write(MT_MAGIC)
        f.write(struct.pack('<i', MT_VERSION))
        f.write(struct.pack('<i', dtype_code))
        f.write(struct.pack('<i', tensor.ndim))
        for dim in tensor.shape:
            f.write(struct.pack('<q', dim))
        f.write(tensor.tobytes())


def load(filepath):
    """
    Load a tensor from .mt format.
    Tries C++ backend first; falls back to pure Python.

    Returns:
        numpy array
    """
    lib = get_lib()
    if lib is not None and hasattr(lib, 'mt_load_paged_mt'):
        lib.mt_load_paged_mt.restype = ctypes.c_void_p
        ptr = lib.mt_load_paged_mt(filepath.encode('utf-8'))
        if ptr:
            lib.mt_get_numel.restype = ctypes.c_int64
            numel = lib.mt_get_numel(ptr)
            lib.mt_get_data_ptr_f32.restype = ctypes.POINTER(ctypes.c_float)
            data_ptr = lib.mt_get_data_ptr_f32(ptr)
            arr = np.ctypeslib.as_array(data_ptr, shape=(numel,)).copy()
            lib.mt_destroy_tensor(ptr)
            return arr

    # Python fallback
    with open(filepath, 'rb') as f:
        magic = f.read(9)
        if magic != MT_MAGIC:
            raise ValueError(f"Invalid .mt file: bad magic {magic!r}")

        version = struct.unpack('<i', f.read(4))[0]
        if version != MT_VERSION:
            raise ValueError(f"Unsupported .mt version {version}")

        dtype_code = struct.unpack('<i', f.read(4))[0]
        ndim = struct.unpack('<i', f.read(4))[0]

        shape = []
        for _ in range(ndim):
            shape.append(struct.unpack('<q', f.read(8))[0])

        dtype = _DTYPE_MAP.get(dtype_code, np.float32)
        data = f.read()
        tensor = np.frombuffer(data, dtype=dtype)

        if shape:
            tensor = tensor.reshape(shape)

        return tensor


def info(filepath):
    """Print metadata for a .mt file without loading the full tensor."""
    with open(filepath, 'rb') as f:
        magic = f.read(9)
        version = struct.unpack('<i', f.read(4))[0]
        dtype_code = struct.unpack('<i', f.read(4))[0]
        ndim = struct.unpack('<i', f.read(4))[0]
        shape = [struct.unpack('<q', f.read(8))[0] for _ in range(ndim)]

    dtype = _DTYPE_MAP.get(dtype_code, np.float32)
    numel = 1
    for s in shape:
        numel *= s
    size_mb = numel * np.dtype(dtype).itemsize / (1024 * 1024)

    print(f"  File:    {filepath}")
    print(f"  Magic:   {magic.decode()}")
    print(f"  Version: {version}")
    print(f"  DType:   {dtype}")
    print(f"  Shape:   {tuple(shape)}")
    print(f"  Numel:   {numel:,}")
    print(f"  Size:    {size_mb:.2f} MB")
