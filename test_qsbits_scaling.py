"""Verify block-scaled QSBits against naive float reference."""

import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mecantensor"))
from mecantensor import qsbits


def reference_scaled_matmul(input_binary, weights_binary, scales, input_scale, group_size):
    """FP32 reference: expand binary to {-1,+1}, apply per-group scales, matmul."""
    batch = input_binary.shape[0]
    K = input_binary.shape[1]
    O = weights_binary.shape[0]
    n_groups = K // group_size
    output = np.zeros((batch, O), dtype=np.float32)

    for b in range(batch):
        for o in range(O):
            acc = 0.0
            for g in range(n_groups):
                start = g * group_size
                end = start + group_size
                x_slice = input_binary[b, start:end].astype(np.float32) * 2.0 - 1.0
                w_slice = weights_binary[o, start:end].astype(np.float32) * 2.0 - 1.0
                dot = np.dot(x_slice, w_slice)
                acc += dot * scales[o, g]
            output[b, o] = acc * input_scale
    return output


def main():
    np.random.seed(42)

    K = 1024
    O = 64
    batch = 4
    group_size = 128
    K_packed = K // 8
    n_groups = K // group_size

    # Generate random binary data (unpacked for reference)
    input_binary = np.random.randint(0, 2, (batch, K), dtype=np.uint8)
    weights_binary = np.random.randint(0, 2, (O, K), dtype=np.uint8)
    scales = np.random.randn(O, n_groups).astype(np.float32) * 0.1 + 1.0
    input_scale = 0.85

    # Pack to uint8
    input_packed = np.packbits(input_binary, axis=1)
    weights_packed = np.packbits(weights_binary, axis=1)

    # Reference (FP32)
    ref_output = reference_scaled_matmul(input_binary, weights_binary, scales, input_scale, group_size)

    # QSBits scaled forward
    qsbits_output = qsbits.forward_scaled(
        input_packed, weights_packed, scales,
        input_scale=input_scale, group_size=group_size
    )

    # Compare
    max_err = np.max(np.abs(ref_output - qsbits_output))
    mean_err = np.mean(np.abs(ref_output - qsbits_output))
    rel_err = np.mean(np.abs(ref_output - qsbits_output) / (np.abs(ref_output) + 1e-8))

    print(f"Shape: batch={batch}, K={K}, O={O}, group_size={group_size}")
    print(f"Max absolute error:  {max_err:.6f}")
    print(f"Mean absolute error: {mean_err:.6f}")
    print(f"Mean relative error: {rel_err:.6f}")

    if max_err < 1e-3:
        print("PASS: Scaled QSBits matches FP32 reference.")
    else:
        print("FAIL: Mismatch detected.")
        sys.exit(1)

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(100):
        qsbits.forward_scaled(input_packed, weights_packed, scales,
                              input_scale=input_scale, group_size=group_size)
    t1 = time.perf_counter()
    ms_per_call = (t1 - t0) / 100 * 1000
    print(f"Benchmark: {ms_per_call:.3f} ms/call (100 iterations, K={K}, O={O})")

    qsbits.info()


if __name__ == "__main__":
    main()
