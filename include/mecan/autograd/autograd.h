#pragma once

#include "variable.h"
#include "function.h"
#include "engine.h"
#include "ops.h"
#include "trace.h"

namespace mecan {
namespace autograd {

    // Higher level API for Variable creation
    inline Variable make_variable(Tensor t, bool requires_grad = false) {
        return Variable(t, requires_grad);
    }

}
}
