#pragma once

#include <cstddef>
#include <cmath>
#include <algorithm>

namespace mecan {
namespace ops {

// Mode 0: Constant (includes zero-pad), 1: Reflect, 2: Replicate
void pad2d(const float* input, float* output,
           size_t N, size_t C, size_t H, size_t W,
           int pad_left, int pad_right, int pad_top, int pad_bottom,
           int mode = 0, float constant_value = 0.0f);

} // namespace ops
} // namespace mecan
