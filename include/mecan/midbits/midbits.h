#pragma once

#include <cstdint>

namespace mecan {
namespace midbits {

void precompute_lut(const float* x_chunk_16, float* lut_256);
void matvec_0_75b(const uint8_t* w_midbits, const float* x, float* y, int m, int k);

} // namespace midbits
} // namespace mecan
