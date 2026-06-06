# MecanTensor v2.1

**Universal Hardware-Accelerated Tensor Engine for ASI Training**

MecanTensor is a from-scratch tensor computation library with a C++ backend and Python interface, featuring three novel sub-bit quantization engines that push beyond the Shannon limit, alongside integrated memory-optimized training components.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import mecantensor as mt
import numpy as np

# Discover all hardware
mt.hal.print_devices()

# Matrix multiplication (routes through C++ HLAS engine)
A = np.random.randn(512, 1024).astype(np.float32)
B = np.random.randn(1024, 256).astype(np.float32)
C = mt.ops.matmul(A, B)

# Save / Load in .mt format
mt.io.save(C, "output.mt")
loaded = mt.io.load("output.mt")
```

### 1. Logical Gradient Compressor (LGC)

LGC is a cache-bounded optimizer that leverages 8-bit gradient quantization, 1-bit sign momentum, and uint8 curvature tracking to reduce training memory overhead.

```python
import mecantensor as mt
import numpy as np

# Define parameters (must have requires_grad=True)
class Parameter:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)
        self.grad = None
        self.requires_grad = True

params = [Parameter(np.random.randn(1024, 1024))]

# Instantiate LGC
# For models > 1B parameters, set total_model_params and provide a cache_dir to enable SSD sharding
optimizer = mt.lgc.LogicalGradientCompressor(
    params,
    lr=1e-3,
    total_model_params=50e9,       # Simulated 50B parameter model
    cache_dir="./optimizer_cache"  # Enable SSD infusion for offloaded states
)

# Training loop step
params[0].grad = np.random.randn(1024, 1024).astype(np.float32)
optimizer.step()

# Clean up states when done
optimizer.cleanup()
```

### 2. SSD Infusion Engine

The Infusion Engine performs selective tensor offload to SSD for models that exceed available RAM, employing memory-mapped files and circular LRU eviction to respect a hard budget ceiling.

```python
import mecantensor as mt
import numpy as np

# Initialize the SSD Infusion Engine
engine = mt.infusion.SSDInfusionEngine(
    cache_directory="./infusion_cache",
    total_model_params=50e9  # Sets budget ceiling (e.g., 10 GB limit for 50B params)
)

# Offload a tensor to disk
tensor_data = np.random.randn(2048, 2048).astype(np.float32)
engine.offload_tensor("layer_12_weights", tensor_data)

# Prefetch and retrieve the tensor back via zero-copy memory-map
engine.hint_prefetch("layer_12_weights")
weights = engine.retrieve_tensor("layer_12_weights")

# Clean up all active memory maps and temporary cache files
engine.cleanup()
```

## Subsystems

| Module | Description | Precision |
|---|---|---|
| `mt.hal` | Hardware Abstraction Layer — probes CPU, GPU, NPU, FPGA | — |
| `mt.hlas` | Hardware Linear Algebra Subroutines (sgemm, add) | FP32 |
| `mt.ops` | Core math (matmul, add, bitlinear, flash attention) | FP32 |
| `mt.fluxbits` | Bloom Tensor Engine — AND + POPCOUNT, no FP multiply | **0.45-bit** |
| `mt.qsbits` | Binary XNOR+POPCOUNT Engine — 64 synapses/clock | **1-bit** |
| `mt.midbits` | Block-Palette LUT Engine — 16-element blocks | **0.75-bit** |
| `mt.io` | `.mt` paged serialization (save/load/stream) | — |
| `mt.tensor` | Low-level C++ tensor bridge via ctypes | — |
| `mt.lgc` | Logical Gradient Compressor (LGC) optimizer | 8-bit Grad, 1-bit Mom |
| `mt.infusion` | SSD Infusion Engine (selective tensor offloader) | — |

## Architecture

```
Python Layer (mecantensor/)
    ├── hal.py        → src/hal/discovery.py (OpenCL/Vulkan/CUDA probe)
    ├── hlas.py       → _native.py → mecantensor_40.dll (C++ HLAS engine)
    ├── ops.py        → _native.py → mecantensor_40.dll (matmul, add, etc)
    ├── fluxbits.py   → Python fallback + C++ (when compiled)
    ├── qsbits.py     → Python fallback + C++ (when compiled)
    ├── midbits.py    → Python fallback + C++ (when compiled)
    ├── io.py         → _native.py → mecantensor_40.dll (.mt format)
    ├── tensor.py     → _native.py → mecantensor_40.dll (raw tensor handle)
    ├── lgc/          → Logical Gradient Compressor optimizer subpackage
    └── infusion/     → SSD Infusion Engine subpackage

C++ Layer (src/)
    ├── hal/          → Hardware discovery & backend registry
    ├── hlas/         → CPU/GPU/NPU/Hybrid linear algebra engines
    ├── fluxbits/     → Bloom tensor compiler & AVX2 kernel
    ├── qsbits/       → XNOR + POPCOUNT binary engine
    ├── midbits/      → 0.75-bit LUT encoder & decoder
    ├── ops/          → math, attention, bitlinear, conv, TSI
    ├── io/           → .mt paged serialization
    └── autograd/     → Tape-based automatic differentiation
```

## Running Tests

```bash
python test_package.py
```

## License

Proprietary — MecanLabs
