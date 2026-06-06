/**
 * TST Master Header
 * High-performance C++ backend for TSSR Superintelligence.
 */

#pragma once

#include <iostream>
#include "core/types.h"
#include "core/allocator.h"
#include "tensor.h"
#include "ops/math.h"
#include "io/serialization.h"
#include "autograd/autograd.h"
#include "parallel.h"
#include "optim/lgc.h"
#include "nn/linear.h"
#include "nn/loss.h"
#include "nn/ntm.h"
#include "ops/attention.h"
#include "ops/spatial.h"
#include "ops/bitlinear.h"
#include "ops/conv.h"
#include "qsbits/qsbits.h"
#include "midbits/midbits.h"
#include "trainer.h"
#include "vision/core_vision.h"
#include "runtime/descriptors.h"
#include "runtime/interfaces.h"
#include "runtime/pager.h"
#include "runtime/backend_registry.h"
#include "runtime/quant_runtime.h"
#include "runtime/scheduler.h"
#include "distributed/collective.h"

namespace mecan {

    inline void info() {
        std::cout << "TST Lib v2.0 - Active" << std::endl;
    }

}
