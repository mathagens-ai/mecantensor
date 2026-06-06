// MecanTensor: Normalization Operations
// BatchNorm2d, LayerNorm, InstanceNorm2d, GroupNorm
// All NCHW layout, OpenMP parallel, production-grade.

#include "mecan/ops/norm.h"
#include <cmath>
#include <cstring>
#include <vector>

namespace mecan {
namespace ops {

// ─── BatchNorm2d ─────────────────────────────────────────────────────────────
// Inference: y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta
// Training:  compute batch statistics, then normalize

void batch_norm2d(
    const float* input, float* output,
    size_t N, size_t C, size_t H, size_t W,
    const float* gamma, const float* beta,
    const float* running_mean, const float* running_var,
    float eps, bool training)
{
    size_t spatial = H * W;

    if (!training) {
        // Inference mode: use running statistics
        #pragma omp parallel for schedule(static)
        for (int64_t c = 0; c < (int64_t)C; ++c) {
            float mean = running_mean[c];
            float inv_std = 1.0f / std::sqrt(running_var[c] + eps);
            float g = gamma[c];
            float b = beta[c];
            float scale = g * inv_std;
            float shift = b - mean * scale;

            for (size_t n = 0; n < N; ++n) {
                const float* in_ptr = input + (n * C + c) * spatial;
                float* out_ptr = output + (n * C + c) * spatial;

                for (size_t s = 0; s < spatial; ++s) {
                    out_ptr[s] = in_ptr[s] * scale + shift;
                }
            }
        }
    } else {
        // Training mode: compute batch mean/var on the fly
        #pragma omp parallel for schedule(static)
        for (int64_t c = 0; c < (int64_t)C; ++c) {
            // Compute mean
            float mean = 0;
            for (size_t n = 0; n < N; ++n) {
                const float* in_ptr = input + (n * C + c) * spatial;
                for (size_t s = 0; s < spatial; ++s)
                    mean += in_ptr[s];
            }
            mean /= (float)(N * spatial);

            // Compute variance
            float var = 0;
            for (size_t n = 0; n < N; ++n) {
                const float* in_ptr = input + (n * C + c) * spatial;
                for (size_t s = 0; s < spatial; ++s) {
                    float d = in_ptr[s] - mean;
                    var += d * d;
                }
            }
            var /= (float)(N * spatial);

            // Normalize
            float inv_std = 1.0f / std::sqrt(var + eps);
            float g = gamma[c];
            float b = beta[c];

            for (size_t n = 0; n < N; ++n) {
                const float* in_ptr = input + (n * C + c) * spatial;
                float* out_ptr = output + (n * C + c) * spatial;
                for (size_t s = 0; s < spatial; ++s) {
                    out_ptr[s] = g * (in_ptr[s] - mean) * inv_std + b;
                }
            }
        }
    }
}

// ─── LayerNorm ───────────────────────────────────────────────────────────────
// Normalizes across the last `norm_size` elements for each of `outer` slices

void layer_norm(
    const float* input, float* output,
    size_t outer, size_t norm_size,
    const float* gamma, const float* beta,
    float eps)
{
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)outer; ++i) {
        const float* in_ptr = input + i * norm_size;
        float* out_ptr = output + i * norm_size;

        // Mean
        float mean = 0;
        for (size_t j = 0; j < norm_size; ++j) mean += in_ptr[j];
        mean /= (float)norm_size;

        // Variance
        float var = 0;
        for (size_t j = 0; j < norm_size; ++j) {
            float d = in_ptr[j] - mean;
            var += d * d;
        }
        var /= (float)norm_size;

        float inv_std = 1.0f / std::sqrt(var + eps);

        for (size_t j = 0; j < norm_size; ++j) {
            out_ptr[j] = gamma[j] * (in_ptr[j] - mean) * inv_std + beta[j];
        }
    }
}

// ─── InstanceNorm2d ──────────────────────────────────────────────────────────
// Normalize each (n, c) slice independently

void instance_norm2d(
    const float* input, float* output,
    size_t N, size_t C, size_t H, size_t W,
    const float* gamma, const float* beta,
    float eps)
{
    size_t spatial = H * W;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < (int64_t)N; ++n) {
        for (int64_t c = 0; c < (int64_t)C; ++c) {
            const float* in_ptr = input + (n * C + c) * spatial;
            float* out_ptr = output + (n * C + c) * spatial;

            float mean = 0;
            for (size_t s = 0; s < spatial; ++s) mean += in_ptr[s];
            mean /= (float)spatial;

            float var = 0;
            for (size_t s = 0; s < spatial; ++s) {
                float d = in_ptr[s] - mean;
                var += d * d;
            }
            var /= (float)spatial;

            float inv_std = 1.0f / std::sqrt(var + eps);
            float g = gamma ? gamma[c] : 1.0f;
            float b = beta ? beta[c] : 0.0f;

            for (size_t s = 0; s < spatial; ++s) {
                out_ptr[s] = g * (in_ptr[s] - mean) * inv_std + b;
            }
        }
    }
}

// ─── GroupNorm ────────────────────────────────────────────────────────────────
// Normalize across groups of channels

void group_norm(
    const float* input, float* output,
    size_t N, size_t C, size_t H, size_t W,
    int num_groups,
    const float* gamma, const float* beta,
    float eps)
{
    size_t spatial = H * W;
    size_t channels_per_group = C / num_groups;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < (int64_t)N; ++n) {
        for (int64_t g = 0; g < (int64_t)num_groups; ++g) {
            size_t c_start = g * channels_per_group;
            size_t group_size = channels_per_group * spatial;

            // Compute group mean
            float mean = 0;
            for (size_t c = c_start; c < c_start + channels_per_group; ++c) {
                const float* in_ptr = input + (n * C + c) * spatial;
                for (size_t s = 0; s < spatial; ++s) mean += in_ptr[s];
            }
            mean /= (float)group_size;

            // Compute group variance
            float var = 0;
            for (size_t c = c_start; c < c_start + channels_per_group; ++c) {
                const float* in_ptr = input + (n * C + c) * spatial;
                for (size_t s = 0; s < spatial; ++s) {
                    float d = in_ptr[s] - mean;
                    var += d * d;
                }
            }
            var /= (float)group_size;

            float inv_std = 1.0f / std::sqrt(var + eps);

            // Normalize
            for (size_t c = c_start; c < c_start + channels_per_group; ++c) {
                const float* in_ptr = input + (n * C + c) * spatial;
                float* out_ptr = output + (n * C + c) * spatial;
                float g_val = gamma ? gamma[c] : 1.0f;
                float b_val = beta ? beta[c] : 0.0f;
                for (size_t s = 0; s < spatial; ++s) {
                    out_ptr[s] = g_val * (in_ptr[s] - mean) * inv_std + b_val;
                }
            }
        }
    }
}

} // namespace ops
} // namespace mecan
