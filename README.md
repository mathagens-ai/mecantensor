<p align="center">
  <h1 align="center">MecanTensor v2.0</h1>
  <p align="center"><strong>Universal Hardware-Accelerated Tensor Engine with Sub-Bit Quantization</strong></p>
</p>

<p align="center">
  <a href="https://pypi.org/project/mecantensor/"><img src="https://img.shields.io/pypi/v/mecantensor?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/mecantensor/"><img src="https://img.shields.io/pypi/pyversions/mecantensor" alt="Python"></a>
  <a href="https://github.com/mathagens-ai/mecantensor/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
</p>

---

MecanTensor is a C++/Python tensor engine designed for training and inference at extreme efficiency. It provides hardware-accelerated math operations, sub-bit weight quantization (0.45-bit, 0.75-bit, 1-bit), automatic differentiation, and a complete vision pipeline — all through a unified Python API backed by a high-performance C++ core.

## Features

| Subsystem | Description |
|-----------|-------------|
| **HAL** | Hardware Abstraction Layer — auto-detects CPU (AVX2/SSE), GPU (CUDA/OpenCL), and NPU backends |
| **HLAS** | Hardware Linear Algebra Subroutines — optimized matmul, GEMM dispatched per hardware |
| **FluxBits** | 0.45-bit Bloom Tensor Engine — hash-based weight compression via Bloom filters |
| **QSBits** | 1-bit Binary Engine — XNOR + POPCOUNT with per-group FP32 scales (1BF16) |
| **MidBits** | 0.75-bit Block-Palette LUT Engine — codebook-based sub-byte quantization |
| **Ops** | Core operations — matmul, conv2d, attention, pooling, normalization, upsampling |
| **Autograd** | Reverse-mode automatic differentiation with operation tracing |
| **Vision** | Detection, color analysis, motion tracking, object recognition, lighting estimation |
| **IO** | Paged `.mt` tensor serialization for efficient model checkpointing |

## Installation

```bash
pip install mecantensor
```

**From source:**
```bash
git clone https://github.com/mathagens-ai/mecantensor.git
cd mecantensor
pip install -e .
```

## Quick Start

```python
import mecantensor as mt

# Hardware discovery
devices = mt.hal.discover()
print(f"Available backends: {devices}")

# Tensor creation and operations
a = mt.tensor.create([4, 256], dtype="float32")
b = mt.tensor.create([256, 128], dtype="float32")
c = mt.ops.matmul(a, b)

# 1-bit quantized forward pass (QSBits)
packed, scales = mt.qsbits.quantize(weight_matrix, group_size=128)
output = mt.qsbits.forward_scaled(input_packed, packed, scales,
                                   input_scale=0.1, group_size=128)

# Save / Load tensors
mt.io.save(c, "output.mt")
loaded = mt.io.load("output.mt")
```

## Building the C++ Engine

The native C++ backend provides AVX2-accelerated operations. To build from source:

**Windows (MSVC):**
```bash
build_dll.bat
```

**CMake (cross-platform):**
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## Architecture

```
┌────────────────────────────────────────────────────┐
│                   Python API                        │
│  hal · hlas · ops · fluxbits · qsbits · midbits    │
│  autograd · vision · io · tensor                    │
├────────────────────────────────────────────────────┤
│               C++ Native Engine                     │
│  src/ops/    src/nn/    src/autograd/              │
│  src/hal/    src/hlas/  src/runtime/               │
│  src/vision/ src/io/    src/fluxbits/              │
│  src/qsbits/ src/midbits/ src/distributed/         │
├────────────────────────────────────────────────────┤
│            Hardware Backends                        │
│  CPU (AVX2/SSE) · GPU (CUDA/OpenCL) · NPU         │
└────────────────────────────────────────────────────┘
```

## Running Tests

```bash
python test_package.py
python test_bench_qsbits.py
python test_qsbits_scaling.py
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

Copyright 2025-2026 Mathagens AI
