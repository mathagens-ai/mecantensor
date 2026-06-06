#pragma once

#include "variable.h"

namespace mecan {
namespace autograd {
namespace functional {

    Variable add(Variable& a, Variable& b);
    Variable matmul(Variable& a, Variable& b);
    Variable relu(Variable& x);

} // namespace functional
} // namespace autograd
} // namespace mecan
