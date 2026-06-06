#pragma once
// MecanTensor: Upsampling and Transposed Convolutions
// Operations essential for Generative AI (GANs, Diffusion models):
// ConvTranspose2d, PixelShuffle, PixelUnshuffle, Interpolate (Nearest/Bilinear).

#include <cstddef>

namespace mecan {
namespace ops {



    // ─── PixelShuffle (Sub-Pixel Convolution) ────────────────────────────────
    // Rearranges elements in a tensor of shape [*, C*r^2, H, W]
    // to a tensor of shape [*, C, H*r, W*r], where r is the upscale factor.
    void pixel_shuffle(
        const float* input, float* output,
        size_t N, size_t C_in, size_t H, size_t W,
        int upscale_factor
    );

    // ─── PixelUnshuffle ──────────────────────────────────────────────────────
    // Reverse of PixelShuffle.
    // Rearranges [*, C, H*r, W*r] -> [*, C*r^2, H, W]
    void pixel_unshuffle(
        const float* input, float* output,
        size_t N, size_t C_in, size_t H, size_t W,
        int downscale_factor
    );

    // ─── Interpolate (Nearest / Bilinear) ────────────────────────────────────
    void upsample_nearest2d(
        const float* input, float* output,
        size_t N, size_t C, size_t H, size_t W,
        size_t out_H, size_t out_W
    );

    void upsample_bilinear2d(
        const float* input, float* output,
        size_t N, size_t C, size_t H, size_t W,
        size_t out_H, size_t out_W,
        bool align_corners
    );

} // namespace ops
} // namespace mecan
