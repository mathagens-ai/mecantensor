"""
Native C/C++ library loader for MecanTensor.

Locates the compiled DLL (mecantensor_40.dll) or shared library
and loads it via ctypes, exposing the raw C bridge functions.

All subsystem wrappers (fluxbits, qsbits, etc.) import from here.
"""

import ctypes
import os
import sys
import platform

_lib = None
_lib_path = None


def _find_native_lib():
    """Search for the compiled MecanTensor native library."""
    # The mecantensor source root (one level up from this file)
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(pkg_dir)

    candidates = []

    if platform.system() == "Windows":
        names = ["mecantensor_40.dll", "mecantensor.dll", "libmecantensor.dll"]
    else:
        names = ["libmecantensor.so", "libmecantensor_40.so"]

    # Search order: root_dir, pkg_dir, build/, PATH
    search_dirs = [
        root_dir,
        pkg_dir,
        os.path.join(root_dir, "build"),
        os.path.join(root_dir, "build", "Release"),
        os.path.join(root_dir, "build", "Debug"),
        sys.prefix, # Try searching in the python installation dir if needed
        os.path.dirname(sys.executable),
    ]

    for d in search_dirs:
        for name in names:
            path = os.path.join(d, name)
            if os.path.isfile(path):
                candidates.append(path)

    # Also search for .pyd files (Python extensions)
    for d in search_dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".pyd") or f.endswith(".so"):
                    candidates.append(os.path.join(d, f))

    return candidates


def _setup_signatures(lib):
    """Set up the argument and return types for pointer-returning and accepting functions."""
    void_p = ctypes.c_void_p
    char_p = ctypes.c_char_p
    int64_p = ctypes.POINTER(ctypes.c_int64)
    int_t = ctypes.c_int
    float_t = ctypes.c_float
    uint64_t = ctypes.c_uint64

    # Tensor management
    if hasattr(lib, 'mt_create_tensor'):
        lib.mt_create_tensor.restype = void_p
        lib.mt_create_tensor.argtypes = [int64_p, int_t, int_t, int_t]
    if hasattr(lib, 'mt_destroy_tensor'):
        lib.mt_destroy_tensor.restype = None
        lib.mt_destroy_tensor.argtypes = [void_p]
    if hasattr(lib, 'mt_get_data_ptr_f32'):
        lib.mt_get_data_ptr_f32.restype = void_p
        lib.mt_get_data_ptr_f32.argtypes = [void_p]
    if hasattr(lib, 'mt_get_data_ptr_u8'):
        lib.mt_get_data_ptr_u8.restype = void_p
        lib.mt_get_data_ptr_u8.argtypes = [void_p]
    if hasattr(lib, 'mt_get_numel'):
        lib.mt_get_numel.restype = ctypes.c_int64
        lib.mt_get_numel.argtypes = [void_p]

    # Core ops
    if hasattr(lib, 'mt_op_add'):
        lib.mt_op_add.restype = None
        lib.mt_op_add.argtypes = [void_p, void_p, void_p]
    if hasattr(lib, 'mt_op_matmul'):
        lib.mt_op_matmul.restype = None
        lib.mt_op_matmul.argtypes = [void_p, void_p, void_p]

    # Serialization
    if hasattr(lib, 'mt_save_paged_mt'):
        lib.mt_save_paged_mt.restype = None
        lib.mt_save_paged_mt.argtypes = [void_p, char_p, char_p, int_t, uint64_t]
    if hasattr(lib, 'mt_load_paged_mt'):
        lib.mt_load_paged_mt.restype = void_p
        lib.mt_load_paged_mt.argtypes = [char_p]

    # Attention & Linear
    if hasattr(lib, 'mt_flash_ternary_attention'):
        lib.mt_flash_ternary_attention.restype = None
        lib.mt_flash_ternary_attention.argtypes = [void_p, void_p, void_p, void_p]
    if hasattr(lib, 'mt_op_bitlinear'):
        lib.mt_op_bitlinear.restype = None
        lib.mt_op_bitlinear.argtypes = [void_p, void_p, void_p, void_p]
    if hasattr(lib, 'mt_op_qsbits'):
        lib.mt_op_qsbits.restype = None
        lib.mt_op_qsbits.argtypes = [void_p, void_p, void_p]
    if hasattr(lib, 'mt_op_qsbits_scaled'):
        lib.mt_op_qsbits_scaled.restype = None
        lib.mt_op_qsbits_scaled.argtypes = [void_p, void_p, void_p, float_t, int_t, void_p]

    # Pools & spatial
    if hasattr(lib, 'mt_op_max_pool2d'):
        lib.mt_op_max_pool2d.restype = None
        lib.mt_op_max_pool2d.argtypes = [void_p, void_p, int_t, int_t, int_t, int_t]
    if hasattr(lib, 'mt_op_avg_pool2d'):
        lib.mt_op_avg_pool2d.restype = None
        lib.mt_op_avg_pool2d.argtypes = [void_p, void_p, int_t, int_t, int_t, int_t]
    if hasattr(lib, 'mt_op_pixel_shuffle'):
        lib.mt_op_pixel_shuffle.restype = None
        lib.mt_op_pixel_shuffle.argtypes = [void_p, void_p, int_t]
    if hasattr(lib, 'mt_op_pad2d'):
        lib.mt_op_pad2d.restype = None
        lib.mt_op_pad2d.argtypes = [void_p, void_p, int_t, int_t, int_t, int_t, int_t, float_t]
    if hasattr(lib, 'mt_op_conv1d'):
        lib.mt_op_conv1d.restype = None
        lib.mt_op_conv1d.argtypes = [void_p, void_p, void_p, int_t, int_t]

    # Vision engine
    if hasattr(lib, 'mt_vision_canny'):
        lib.mt_vision_canny.restype = None
        lib.mt_vision_canny.argtypes = [void_p, void_p, float_t, float_t]
    if hasattr(lib, 'mt_vision_farneback_flow'):
        lib.mt_vision_farneback_flow.restype = None
        lib.mt_vision_farneback_flow.argtypes = [void_p, void_p, void_p, int_t, int_t]
    if hasattr(lib, 'mt_vision_signed_distance_field'):
        lib.mt_vision_signed_distance_field.restype = None
        lib.mt_vision_signed_distance_field.argtypes = [void_p, void_p]
    if hasattr(lib, 'mt_vision_diffraction_pattern'):
        lib.mt_vision_diffraction_pattern.restype = None
        lib.mt_vision_diffraction_pattern.argtypes = [void_p, void_p, float_t, float_t, float_t, float_t]


def get_lib():
    """Get the loaded ctypes CDLL handle. Loads on first call."""
    global _lib, _lib_path

    if _lib is not None:
        return _lib

    candidates = _find_native_lib()

    # Try the main DLL first
    for path in candidates:
        if "mecantensor_40" in os.path.basename(path) and path.endswith(".dll"):
            try:
                _lib = ctypes.CDLL(path)
                _lib_path = path
                _setup_signatures(_lib)
                return _lib
            except OSError:
                continue

    # Try any DLL/SO
    for path in candidates:
        if path.endswith(".dll") or path.endswith(".so"):
            try:
                _lib = ctypes.CDLL(path)
                _lib_path = path
                _setup_signatures(_lib)
                return _lib
            except OSError:
                continue

    # No native lib found — return None (wrappers will use fallbacks)
    return None


def is_available():
    """Check if the native C++ backend is loaded."""
    return get_lib() is not None


def lib_path():
    """Return the path to the loaded native library, or None."""
    get_lib()  # ensure attempted
    return _lib_path
