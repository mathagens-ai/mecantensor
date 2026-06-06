"""
MecanTensor v2.0 — Full Package Validation Test
Tests every subsystem: HAL, HLAS, FluxBits, QSBits, MidBits, IO, Ops, Tensor
"""
import numpy as np
import os
import sys

PASS = 0
FAIL = 0

def test(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")

print("=" * 60)
print("  MECANTENSOR v2.0 — PACKAGE VALIDATION")
print("=" * 60)

# ─── 1. Import Test ───
print("\n--- 1. IMPORT TEST ---")
import mecantensor as mt
test("import mecantensor", True)
test("__version__ exists", hasattr(mt, '__version__'))
test("version = 2.0.0", mt.__version__ == "2.0.0")
test("hal submodule", hasattr(mt, 'hal'))
test("hlas submodule", hasattr(mt, 'hlas'))
test("fluxbits submodule", hasattr(mt, 'fluxbits'))
test("qsbits submodule", hasattr(mt, 'qsbits'))
test("midbits submodule", hasattr(mt, 'midbits'))
test("ops submodule", hasattr(mt, 'ops'))
test("io submodule", hasattr(mt, 'io'))
test("tensor submodule", hasattr(mt, 'tensor'))

# ─── 2. HAL Test ───
print("\n--- 2. HAL DISCOVERY ---")
devices = mt.hal.discover()
test("HAL discover() returns list", isinstance(devices, list))
test("At least 1 device found", len(devices) >= 1)

# ─── 3. Native DLL Test ───
print("\n--- 3. NATIVE C++ BRIDGE ---")
from mecantensor._native import is_available, lib_path
test("DLL loaded", is_available())
test("DLL path resolved", lib_path() is not None)
if lib_path():
    print(f"       DLL: {lib_path()}")

# ─── 4. Ops Test (matmul + add) ───
print("\n--- 4. OPS (matmul, add) ---")
A = np.random.randn(64, 128).astype(np.float32)
B = np.random.randn(128, 32).astype(np.float32)
C = mt.ops.matmul(A, B)
test("matmul shape", C.shape == (64, 32))
C_ref = A @ B
test("matmul accuracy", np.allclose(C, C_ref, atol=1e-3))

X = np.random.randn(100).astype(np.float32)
Y = np.random.randn(100).astype(np.float32)
Z = mt.ops.add(X, Y)
test("add shape", Z.shape == (100,))
test("add accuracy", np.allclose(Z, X + Y, atol=1e-6))

# ─── 5. HLAS Test ───
print("\n--- 5. HLAS (sgemm) ---")
C2 = mt.hlas.sgemm(A, B)
test("hlas.sgemm shape", C2.shape == (64, 32))
test("hlas.sgemm accuracy", np.allclose(C2, C_ref, atol=1e-3))

# ─── 6. FluxBits Test ───
print("\n--- 6. FLUXBITS (0.45-bit) ---")
dense = np.random.randn(16, 64).astype(np.float32)
compiled = mt.fluxbits.compile_weights(dense)
test("compile returns dict", isinstance(compiled, dict))
test("flux_rows shape", compiled['flux_rows'].shape[0] == 16)
test("density > 0", compiled['density'] > 0)
print(f"       Density: {compiled['density']:.3f}")

# Forward pass
input_bin = np.random.randint(0, 255, (2, compiled['K_bytes']), dtype=np.uint8)
output = mt.fluxbits.forward(input_bin, compiled)
test("forward shape", output.shape == (2, 16))
test("forward not all zero", np.any(output != 0))

# ─── 7. QSBits Test ───
print("\n--- 7. QSBITS (1-bit) ---")
inp = np.random.randn(4, 128).astype(np.float32)
inp_packed = mt.qsbits.pack_binary(inp[0])
test("pack_binary", inp_packed.dtype == np.uint8)

weights_packed = np.random.randint(0, 255, (8, inp_packed.shape[0]), dtype=np.uint8)
out = mt.qsbits.forward(inp_packed, weights_packed)
test("qsbits forward shape", out.shape == (1, 8))
test("qsbits forward not zero", np.any(out != 0))

# ─── 8. IO Test (.mt format) ───
print("\n--- 8. IO (.mt serialization) ---")
test_tensor = np.random.randn(32, 64).astype(np.float32)
test_path = os.path.join(os.path.dirname(__file__), "_test_output.mt")
mt.io.save(test_tensor, test_path)
test(".mt file created", os.path.exists(test_path))

loaded = mt.io.load(test_path)
test(".mt load shape", loaded.shape == (32, 64) or loaded.size == 32 * 64)
test(".mt roundtrip accuracy", np.allclose(loaded.flatten(), test_tensor.flatten(), atol=1e-6))

# Cleanup
os.remove(test_path)
test(".mt cleanup", not os.path.exists(test_path))

# ─── 9. MidBits Test ───
print("\n--- 9. MIDBITS (0.75-bit) ---")
x_chunk = np.random.randn(16).astype(np.float32)
lut = mt.midbits.precompute_lut(x_chunk)
test("LUT shape", lut.shape == (256,))
test("LUT not all zero", np.any(lut != 0))

# ─── Summary ───
print("\n" + "=" * 60)
total = PASS + FAIL
print(f"  RESULTS: {PASS}/{total} passed, {FAIL} failed")
if FAIL == 0:
    print("  STATUS: ALL TESTS PASSED")
else:
    print(f"  STATUS: {FAIL} FAILURE(S)")
print("=" * 60)
