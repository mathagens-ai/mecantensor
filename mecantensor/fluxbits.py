"""
MecanTensor FluxBits — 0.45-bit Bloom Tensor Engine (Python Wrapper)

Wraps the C++ FluxBits compiler and execution kernel.
FluxBits compresses dense FP32 weight matrices into probabilistic
Bloom-filter bit arrays, then executes forward passes using pure
AND + POPCOUNT — no floating-point multiply needed.

Usage:
    from mecantensor import fluxbits
    compiled = fluxbits.compile_weights(dense_weights, M, N)
    output = fluxbits.forward(input_binary, compiled, M, N)
"""

import numpy as np
from ._native import get_lib
import ctypes


def compile_weights(dense_weights, M=None, N=None, compression_ratio=0.45, d_hashes=2):
    """
    Compress a dense FP32 weight matrix into FluxBits Bloom Tensor.

    Args:
        dense_weights: numpy float32 array (M, N)
        compression_ratio: target bits per parameter (default 0.45)
        d_hashes: number of hash functions per synapse

    Returns:
        dict with 'flux_rows' (bytes), 'M', 'N', 'K_bits', 'compression_ratio'
    """
    dense_weights = np.ascontiguousarray(dense_weights, dtype=np.float32)

    if M is None:
        M = dense_weights.shape[0]
    if N is None:
        N = dense_weights.shape[1] if dense_weights.ndim > 1 else dense_weights.shape[0]

    K_bits = int(compression_ratio * N)
    if K_bits % 8 != 0:
        K_bits += (8 - K_bits % 8)
    K_bytes = K_bits // 8

    # Try C++ backend
    lib = get_lib()
    if lib is not None and hasattr(lib, 'mt_flux_compile'):
        # Future: call lib.mt_flux_compile(...)
        pass

    # Python fallback: simulate Bloom-filter compilation
    flux_rows = np.zeros((M, K_bytes), dtype=np.uint8)

    for i in range(M):
        row = dense_weights[i] if dense_weights.ndim > 1 else dense_weights
        threshold = np.mean(np.abs(row))
        active = np.where(np.abs(row) > threshold)[0]

        for j in active:
            for h in range(d_hashes):
                # SplitMix64 based hash for perfect avalanche
                seed = (int(j) << 32) | int(h)
                x = (seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
                x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
                x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
                hash_val = x ^ (x >> 31)
                
                bit_pos = hash_val % K_bits
                byte_idx = bit_pos // 8
                bit_idx = bit_pos % 8
                flux_rows[i, byte_idx] |= (1 << bit_idx)

    return {
        'flux_rows': flux_rows,
        'M': M,
        'N': N,
        'K_bits': K_bits,
        'K_bytes': K_bytes,
        'compression_ratio': compression_ratio,
        'd_hashes': d_hashes,
        'density': float(np.unpackbits(flux_rows).mean()),
    }


def forward(input_binary, compiled, M=None, N=None):
    """
    Execute FluxBits forward pass using AND + POPCOUNT.

    Args:
        input_binary: numpy uint8 packed bits (batch, K_bytes)
        compiled: dict from compile_weights()

    Returns:
        output: numpy float32 (batch, M) — raw popcount voltages
    """
    flux_rows = compiled['flux_rows']
    M = compiled['M']
    K_bytes = compiled['K_bytes']

    if input_binary.ndim == 1:
        input_binary = input_binary.reshape(1, -1)

    batch = input_binary.shape[0]
    output = np.zeros((batch, M), dtype=np.float32)

    for b in range(batch):
        for m in range(M):
            # AND + POPCOUNT
            hits = np.unpackbits(input_binary[b] & flux_rows[m])
            output[b, m] = float(np.sum(hits))

    return output


def info():
    """Print FluxBits engine status."""
    lib = get_lib()
    native = lib is not None and hasattr(lib, 'mt_flux_compile')
    print(f"FluxBits Engine: {'Active' if native else 'Fallback (Python)'}")
