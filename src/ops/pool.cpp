// MecanTensor: Pooling Operations — AVX2-Ready Implementation
// MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d
// All support NCHW layout with OpenMP parallelism.

#include "mecan/ops/pool.h"
#include <cmath>
#include <algorithm>
#include <limits>

#ifdef _MSC_VER
#include <immintrin.h>
#endif

namespace mecan {
namespace ops {

// ─── MaxPool2d ───────────────────────────────────────────────────────────────

void max_pool2d(
    const float* input, float* output,
    size_t N, size_t C, size_t H, size_t W,
    int kH, int kW, int stride, int pad)
{
    size_t H_out = (H + 2*pad - kH) / stride + 1;
    size_t W_out = (W + 2*pad - kW) / stride + 1;

    #pragma omp parallel for collapse(3) schedule(static)
    for (int64_t n = 0; n < (int64_t)N; ++n) {
        for (int64_t c = 0; c < (int64_t)C; ++c) {
            for (int64_t oh = 0; oh < (int64_t)H_out; ++oh) {
                const float* in_nc = input + (n * C + c) * H * W;
                float* out_nc = output + (n * C + c) * H_out * W_out;

                for (size_t ow = 0; ow < W_out; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();

                    for (int kh = 0; kh < kH; ++kh) {
                        for (int kw = 0; kw < kW; ++kw) {
                            int ih = (int)(oh * stride + kh) - pad;
                            int iw = (int)(ow * stride + kw) - pad;
                            if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                                float val = in_nc[ih * W + iw];
                                if (val > max_val) max_val = val;
                            }
                        }
                    }
                    out_nc[oh * W_out + ow] = max_val;
                }
            }
        }
    }
}

// ─── AvgPool2d ───────────────────────────────────────────────────────────────

void avg_pool2d(
    const float* input, float* output,
    size_t N, size_t C, size_t H, size_t W,
    int kH, int kW, int stride, int pad)
{
    size_t H_out = (H + 2*pad - kH) / stride + 1;
    size_t W_out = (W + 2*pad - kW) / stride + 1;

    #pragma omp parallel for collapse(3) schedule(static)
    for (int64_t n = 0; n < (int64_t)N; ++n) {
        for (int64_t c = 0; c < (int64_t)C; ++c) {
            for (int64_t oh = 0; oh < (int64_t)H_out; ++oh) {
                const float* in_nc = input + (n * C + c) * H * W;
                float* out_nc = output + (n * C + c) * H_out * W_out;

                for (size_t ow = 0; ow < W_out; ++ow) {
                    float sum = 0;
                    int count = 0;

                    for (int kh = 0; kh < kH; ++kh) {
                        for (int kw = 0; kw < kW; ++kw) {
                            int ih = (int)(oh * stride + kh) - pad;
                            int iw = (int)(ow * stride + kw) - pad;
                            if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                                sum += in_nc[ih * W + iw];
                                count++;
                            }
                        }
                    }
                    out_nc[oh * W_out + ow] = (count > 0) ? sum / count : 0.0f;
                }
            }
        }
    }
}

// ─── AdaptiveAvgPool2d ───────────────────────────────────────────────────────
// PyTorch-compatible: automatically computes window boundaries for each output pixel

void adaptive_avg_pool2d(
    const float* input, float* output,
    size_t N, size_t C, size_t H, size_t W,
    size_t out_H, size_t out_W)
{
    #pragma omp parallel for collapse(3) schedule(static)
    for (int64_t n = 0; n < (int64_t)N; ++n) {
        for (int64_t c = 0; c < (int64_t)C; ++c) {
            for (int64_t oh = 0; oh < (int64_t)out_H; ++oh) {
                const float* in_nc = input + (n * C + c) * H * W;
                float* out_nc = output + (n * C + c) * out_H * out_W;

                // PyTorch adaptive pooling: floor-based bin boundaries
                size_t ih_start = (oh * H) / out_H;
                size_t ih_end   = ((oh + 1) * H) / out_H;

                for (size_t ow = 0; ow < out_W; ++ow) {
                    size_t iw_start = (ow * W) / out_W;
                    size_t iw_end   = ((ow + 1) * W) / out_W;

                    float sum = 0;
                    int count = 0;
                    for (size_t ih = ih_start; ih < ih_end; ++ih) {
                        for (size_t iw = iw_start; iw < iw_end; ++iw) {
                            sum += in_nc[ih * W + iw];
                            count++;
                        }
                    }
                    out_nc[oh * out_W + ow] = (count > 0) ? sum / count : 0.0f;
                }
            }
        }
    }
}

// ─── AdaptiveMaxPool2d ───────────────────────────────────────────────────────

void adaptive_max_pool2d(
    const float* input, float* output,
    size_t N, size_t C, size_t H, size_t W,
    size_t out_H, size_t out_W)
{
    #pragma omp parallel for collapse(3) schedule(static)
    for (int64_t n = 0; n < (int64_t)N; ++n) {
        for (int64_t c = 0; c < (int64_t)C; ++c) {
            for (int64_t oh = 0; oh < (int64_t)out_H; ++oh) {
                const float* in_nc = input + (n * C + c) * H * W;
                float* out_nc = output + (n * C + c) * out_H * out_W;

                size_t ih_start = (oh * H) / out_H;
                size_t ih_end   = ((oh + 1) * H) / out_H;

                for (size_t ow = 0; ow < out_W; ++ow) {
                    size_t iw_start = (ow * W) / out_W;
                    size_t iw_end   = ((ow + 1) * W) / out_W;

                    float max_val = -std::numeric_limits<float>::infinity();
                    for (size_t ih = ih_start; ih < ih_end; ++ih) {
                        for (size_t iw = iw_start; iw < iw_end; ++iw) {
                            float v = in_nc[ih * W + iw];
                            if (v > max_val) max_val = v;
                        }
                    }
                    out_nc[oh * out_W + ow] = max_val;
                }
            }
        }
    }
}

} // namespace ops
} // namespace mecan
