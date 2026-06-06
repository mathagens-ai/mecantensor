"""
MecanTensor Tensor — Low-level C++ Tensor Bridge (Python Wrapper)

Provides a Python class that wraps the C++ mecan::Tensor directly
via ctypes. Allows creating, destroying, and accessing tensor data
at the raw pointer level.

Usage:
    from mecantensor.tensor import MecanTensor
    t = MecanTensor([1024, 1024])
    t.fill_random()
    data = t.to_numpy()
"""

import numpy as np
from ._native import get_lib, is_available
import ctypes


class MecanTensor:
    """
    Python handle to a native C++ mecan::Tensor.
    Uses ctypes to bridge into the compiled DLL.
    """

    # ScalarType enum matching C++
    FLOAT32 = 0
    FLOAT64 = 1
    FLOAT16 = 2
    INT32 = 3
    INT8 = 4
    UINT8 = 5

    # DeviceType enum
    CPU = 0
    GPU = 1

    def __init__(self, shape, dtype=0, device=0):
        """
        Create a native MecanTensor.

        Args:
            shape: list/tuple of dimensions
            dtype: ScalarType enum (0=FP32, 4=INT8, etc.)
            device: DeviceType enum (0=CPU, 1=GPU)
        """
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self._ptr = None

        lib = get_lib()
        if lib is None:
            raise RuntimeError(
                "MecanTensor native library not found. "
                "Ensure mecantensor_40.dll is compiled."
            )

        ndim = len(shape)
        c_shape = (ctypes.c_int64 * ndim)(*shape)
        self._ptr = lib.mt_create_tensor(c_shape, ndim, dtype, device)
        self._lib = lib

    def __del__(self):
        if self._ptr is not None and hasattr(self, '_lib') and self._lib is not None:
            try:
                self._lib.mt_destroy_tensor(self._ptr)
            except Exception:
                pass
            self._ptr = None

    @property
    def numel(self):
        """Total number of elements."""
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def data_ptr(self):
        """Get raw float32 pointer to the tensor data."""
        self._lib.mt_get_data_ptr_f32.restype = ctypes.POINTER(ctypes.c_float)
        return self._lib.mt_get_data_ptr_f32(self._ptr)

    def to_numpy(self):
        """Copy tensor data into a numpy array."""
        ptr = self.data_ptr
        arr = np.ctypeslib.as_array(ptr, shape=(self.numel,)).copy()
        return arr.reshape(self.shape)

    def from_numpy(self, arr):
        """Copy numpy data into this tensor."""
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        assert arr.size == self.numel, f"Size mismatch: {arr.size} vs {self.numel}"
        ctypes.memmove(self.data_ptr, arr.ctypes.data, arr.nbytes)

    def fill_random(self, low=-1.0, high=1.0):
        """Fill with uniform random values."""
        data = np.random.uniform(low, high, self.shape).astype(np.float32)
        self.from_numpy(data)

    def fill_zeros(self):
        """Fill with zeros."""
        data = np.zeros(self.shape, dtype=np.float32)
        self.from_numpy(data)

    def __repr__(self):
        backend = "C++" if self._ptr else "None"
        return f"MecanTensor(shape={self.shape}, dtype={self.dtype}, backend={backend})"
