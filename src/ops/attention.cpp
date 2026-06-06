#include "mecan/ops/attention.h"
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mecan {
namespace ops {

    void flash_ternary_attention(const Tensor& query, const Tensor& key_cache, const Tensor& value_cache, Tensor& output) {
        if (query.ndimension() != 2 || key_cache.ndimension() != 2 || value_cache.ndimension() != 2 || output.ndimension() != 2) {
            throw std::runtime_error("flash_ternary_attention expects 2D tensors [S, D].");
        }
        if (query.dtype() != core::ScalarType::Float32 || value_cache.dtype() != core::ScalarType::Float32 ||
            output.dtype() != core::ScalarType::Float32) {
            throw std::runtime_error("flash_ternary_attention expects FP32 query/value/output tensors.");
        }

        size_t seq_len = query.size(0);
        size_t d_k = query.size(1);
        size_t d_v = value_cache.size(1);
        if (key_cache.size(0) != seq_len || key_cache.size(1) != d_k || value_cache.size(0) != seq_len ||
            output.size(0) != seq_len || output.size(1) != d_v) {
            throw std::runtime_error("flash_ternary_attention shape mismatch.");
        }

        const float* q_ptr = query.data_ptr<float>();
        const float* v_ptr = value_cache.data_ptr<float>();
        float* o_ptr = output.data_ptr<float>();
        const bool key_is_int8 = (key_cache.dtype() == core::ScalarType::Int8 || key_cache.dtype() == core::ScalarType::Ternary);
        const bool key_is_fp32 = (key_cache.dtype() == core::ScalarType::Float32);
        if (!key_is_int8 && !key_is_fp32) {
            throw std::runtime_error("flash_ternary_attention key dtype unsupported.");
        }
        const int8_t* k_i8 = key_is_int8 ? key_cache.data_ptr<int8_t>() : nullptr;
        const float* k_f32 = key_is_fp32 ? key_cache.data_ptr<float>() : nullptr;
        float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

        #pragma omp parallel for
        for (int64_t i = 0; i < (int64_t)seq_len; ++i) {
            std::vector<float> scores(seq_len, 0.0f);
            float max_score = -std::numeric_limits<float>::infinity();

            for (int64_t j = 0; j < (int64_t)seq_len; ++j) {
                float score = 0.0f;
                for (int64_t d = 0; d < (int64_t)d_k; ++d) {
                    const float q = q_ptr[i * d_k + d];
                    if (key_is_int8) {
                        const int8_t k_val = k_i8[j * d_k + d];
                        score += q * static_cast<float>(k_val);
                    } else {
                        const float k = k_f32[j * d_k + d];
                        score += q * k;
                    }
                }
                score *= scale;
                scores[j] = score;
                max_score = std::max(max_score, score);
            }

            float denom = 0.0f;
            for (int64_t j = 0; j < (int64_t)seq_len; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                denom += scores[j];
            }
            if (denom <= 0.0f) {
                denom = 1.0f;
            }

            for (int64_t d = 0; d < (int64_t)d_v; ++d) {
                float acc = 0.0f;
                for (int64_t j = 0; j < (int64_t)seq_len; ++j) {
                    const float p = scores[j] / denom;
                    acc += p * v_ptr[j * d_v + d];
                }
                o_ptr[i * d_v + d] = acc;
            }
        }
    }

} // namespace ops
} // namespace mecan
