"""
MecanTensor MidBits - 0.75-bit Block-Palette LUT Engine (Python Wrapper)

Wraps the C++ MidBits kernel (precompute_lut, matvec_0_75b).
Groups weights into 16-element blocks, encodes each block as a 2-bit
palette index into a precomputed LUT of 256 dot-product results.

Usage:
    from mecantensor import midbits
    output = midbits.matvec(weights_075, x, M, K)
"""

import numpy as np
import os
import sys
from ._native import get_lib

# ─── Locate generator.py robustly ───
_this_dir = os.path.dirname(os.path.abspath(__file__))
_search_paths = [
    os.path.abspath(os.path.join(_this_dir, "..", "src", "midbits")),
    os.path.abspath(os.path.join(_this_dir, "..", "..", "src", "midbits")),
]

for _p in _search_paths:
    if os.path.isfile(os.path.join(_p, "generator.py")) and _p not in sys.path:
        sys.path.insert(0, _p)
        break

try:
    from generator import MidBitsGenerator
    _HAS_GENERATOR = True
except ImportError:
    _HAS_GENERATOR = False
    MidBitsGenerator = None


def precompute_lut(x_chunk_16):
    """
    Precompute the 256-entry dot-product LUT for a 16-element input chunk.

    Args:
        x_chunk_16: numpy float32 array of exactly 16 elements

    Returns:
        lut_256: numpy float32 array of 256 precomputed dot products
    """
    x = np.asarray(x_chunk_16, dtype=np.float32).flatten()[:16]
    assert len(x) == 16, "Input chunk must have exactly 16 elements"

    lut = np.zeros(256, dtype=np.float32)
    palette = np.array([-1.0, -0.33, 0.33, 1.0], dtype=np.float32)

    for idx in range(256):
        val = 0.0
        for p in range(4):
            pal_idx = (idx >> (p * 2)) & 0x3
            val += palette[pal_idx] * x[p]
        lut[idx] = val

    return lut


def matvec(weights_075, x, M, K):
    """
    Matrix-vector multiply using 0.75-bit MidBits encoding.

    Args:
        weights_075: numpy uint8 array (M, K_packed) - 0.75-bit encoded weights
        x: numpy float32 array (K,) - input vector
        M: number of output rows
        K: number of input columns

    Returns:
        y: numpy float32 array (M,) - output vector
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.zeros(M, dtype=np.float32)

    blocks = K // 16
    for m in range(M):
        acc = 0.0
        for blk in range(blocks):
            x_chunk = x[blk * 16: blk * 16 + 16]
            lut = precompute_lut(x_chunk[:4])
            byte_idx = m * (K // 16) + blk
            if byte_idx < len(weights_075.flat):
                lut_idx = int(weights_075.flat[byte_idx])
                acc += lut[lut_idx % 256]
        y[m] = acc

    return y


def get_generator():
    """Return the MidBits codebook generator, if available."""
    if _HAS_GENERATOR:
        return MidBitsGenerator
    return None


def info():
    """Print MidBits engine status."""
    lib = get_lib()
    native = lib is not None and hasattr(lib, 'mt_midbits_matvec')
    print(f"MidBits Engine: {'Active' if native else 'Fallback (Python)'}")
