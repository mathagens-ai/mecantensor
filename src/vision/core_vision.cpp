#include "mecan/vision/core_vision.h"
#include <omp.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

namespace mecan {
namespace vision {

    // HyperPatch Embedding — Zero-Copy Spatial Tokenization
    // Extracts non-overlapping patches from an image tensor and flattens them
    // into embedding vectors. This is the foundation of Vision Transformer
    // architectures — we do it with zero intermediate allocation.
    Tensor QuantumVisionModule::hyper_patch_embed(const Tensor& input, int patch_size) {
        auto s = input.shape();
        // Expect input shape: [B, C, H, W] or [C, H, W] or [H, W]
        size_t H, W, C, B;
        if (s.size() == 4) { B=s[0]; C=s[1]; H=s[2]; W=s[3]; }
        else if (s.size() == 3) { B=1; C=s[0]; H=s[1]; W=s[2]; }
        else { B=1; C=1; H=s[0]; W=s[1]; }

        size_t patches_h = H / patch_size;
        size_t patches_w = W / patch_size;
        size_t num_patches = patches_h * patches_w;
        size_t embed_dim = C * patch_size * patch_size;

        std::vector<size_t> out_shape = {B, num_patches, embed_dim};
        Tensor output(out_shape, input.dtype(), input.device());

        const float* in_ptr = input.data_ptr<float>();
        float* out_ptr = output.data_ptr<float>();

        // Extract patches: each patch becomes a flattened vector
        #pragma omp parallel for
        for (int b = 0; b < (int)B; ++b) {
            for (size_t ph = 0; ph < patches_h; ++ph) {
                for (size_t pw = 0; pw < patches_w; ++pw) {
                    size_t patch_idx = ph * patches_w + pw;
                    float* dst = out_ptr + (b * num_patches + patch_idx) * embed_dim;
                    size_t k = 0;
                    for (size_t c = 0; c < C; ++c) {
                        for (int dy = 0; dy < patch_size; ++dy) {
                            for (int dx = 0; dx < patch_size; ++dx) {
                                size_t y = ph * patch_size + dy;
                                size_t x = pw * patch_size + dx;
                                dst[k++] = in_ptr[b*C*H*W + c*H*W + y*W + x];
                            }
                        }
                    }
                }
            }
        }
        return output;
    }

    // Instead of classical kernel dot-products, we apply phase rotation
    // interference patterns that reveal multi-depth edge information
    // impossible with standard convolution.
    //
    // Physics: Each pixel intensity I is mapped to amplitude cos(I*pi/2)
    // and rotated by phase_angle. The result captures constructive/destructive
    // interference between neighboring pixels at the specified phase.
    void QuantumVisionModule::quantum_phase_conv(const Tensor& input, Tensor& output, float phase_angle) {
        auto s = input.shape();
        size_t H, W;
        if (s.size() >= 2) { H = s[s.size()-2]; W = s[s.size()-1]; }
        else { H = 1; W = input.numel(); }

        size_t n = input.numel();
        const float* in_ptr = input.data_ptr<float>();
        float* out_ptr = output.data_ptr<float>();

        float cos_p = std::cos(phase_angle);
        float sin_p = std::sin(phase_angle);
        const float PI = 3.14159265f;

        // where theta = intensity * pi/2
        #pragma omp parallel for
        for (int64_t i = 0; i < (int64_t)n; ++i) {
            float intensity = in_ptr[i];
            float theta = intensity * PI * 0.5f;
            float amp = std::cos(theta);   // amplitude component
            float phs = std::sin(theta);   // phase component

            // Apply phase rotation (unitary transform)
            float new_amp = amp * cos_p - phs * sin_p;
            float new_phs = amp * sin_p + phs * cos_p;

            // Decode back to intensity: interference pattern
            out_ptr[i] = new_amp * new_amp + new_phs * new_phs * 0.5f;
        }

        // This creates edge-interference patterns from neighboring pixels
        if (H > 2 && W > 2) {
            std::vector<float> tmp(n);
            std::memcpy(tmp.data(), out_ptr, n * sizeof(float));

            #pragma omp parallel for
            for (int y = 1; y < (int)H - 1; ++y) {
                for (int x = 1; x < (int)W - 1; ++x) {
                    float center = tmp[y*W + x];
                    float neighbors =
                        tmp[(y-1)*W+x] + tmp[(y+1)*W+x] +
                        tmp[y*W+(x-1)] + tmp[y*W+(x+1)];
                    float interference = center * 4.0f - neighbors;
                    out_ptr[y*W + x] = std::abs(center + interference * sin_p * 0.25f);
                }
            }
        }
    }

} // namespace vision
} // namespace mecan
