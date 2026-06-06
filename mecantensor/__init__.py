"""
MecanTensor — Hardware-Accelerated Tensor Engine

Subsystems:
    mecantensor.hal       - Hardware Abstraction Layer (device probe)
    mecantensor.hlas      - Hardware Linear Algebra Subroutines
    mecantensor.fluxbits  - 0.45-bit Bloom Tensor Execution Engine
    mecantensor.qsbits    - 1-bit XNOR+POPCOUNT Binary Engine
    mecantensor.midbits   - 0.75-bit Block-Palette LUT Engine
    mecantensor.io        - .mt Paged Serialization (save/load)
    mecantensor.ops       - Core math ops (matmul, add, conv2d, etc)
    mecantensor.tensor    - Low-level C++ tensor bridge (ctypes)
    mecantensor.lgc       - Logical Gradient Compressor (optimizer)
    mecantensor.infusion  - SSD Infusion Engine (offload cache)

Usage:
    import mecantensor as mt

    mt.hal.discover()
    mt.ops.matmul(A, B)
    mt.fluxbits.compile_weights(dense_matrix)
    mt.qsbits.forward(input_packed, weights_packed)
    mt.midbits.matvec(weights_075, x)
"""

__version__ = "2.1.0"
__author__ = "Aryan / MecanLabs"

# ─── Submodule imports (relative) ───
from . import hal
from . import hlas
from . import fluxbits
from . import qsbits
from . import midbits
from . import ops
from . import io
from . import tensor
from . import vision
from . import lgc
from . import infusion

__all__ = [
    "hal",
    "hlas",
    "fluxbits",
    "qsbits",
    "midbits",
    "ops",
    "io",
    "tensor",
    "vision",
    "lgc",
    "infusion",
]
