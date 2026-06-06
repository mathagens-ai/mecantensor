#pragma once
// Low-level edge and corner detection kernels. No pretrained models needed.
// All operations work on single-channel float* grayscale images in HW layout.

#include <cstddef>
#include <cstdint>

namespace mecan {
namespace vision {
namespace detect {

    // ─── Edge Detection ──────────────────────────────────────────────────────

    // 1. Sobel gradient
    //    axis: 0=horizontal (dx), 1=vertical (dy), 2=magnitude (sqrt(dx²+dy²))
    void sobel(const float* gray, float* out, size_t H, size_t W, int axis);

    // 2. Canny edge detection (full pipeline)
    //    Gaussian blur → Sobel gradient → Non-max suppression → Hysteresis
    void canny(const float* gray, float* edges, size_t H, size_t W,
               float low_threshold, float high_threshold);

    // 3. Laplacian of Gaussian
    void laplacian(const float* gray, float* out, size_t H, size_t W);

    // ─── Corner Detection ────────────────────────────────────────────────────

    // 4. Harris corner response
    //    output: corner response map R = det(M) - k * trace(M)^2
    void harris_corners(const float* gray, float* response, size_t H, size_t W,
                        float k, int block_size);

    //    output: min eigenvalue map (good features to track)
    void shi_tomasi(const float* gray, float* response, size_t H, size_t W,
                    int block_size);

    // ─── Utility ─────────────────────────────────────────────────────────────

    // 6. Gaussian blur (used internally by Canny and others)
    void gaussian_blur(const float* in, float* out, size_t H, size_t W,
                       int kernel_size, float sigma);

    // Non-maximum suppression for edge thinning (used by Canny)
    void non_max_suppression(const float* magnitude, const float* direction,
                             float* out, size_t H, size_t W);

    // Hysteresis thresholding (used by Canny)
    void hysteresis_threshold(const float* thin_edges, float* out,
                              size_t H, size_t W,
                              float low_thresh, float high_thresh);

} // namespace detect
} // namespace vision
} // namespace mecan
