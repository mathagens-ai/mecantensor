#pragma once

#include "mecan/tensor.h"
#include <cstddef>
#include <vector>

namespace mecan {
namespace vision {
namespace motion {

    /**
     * Optical Flow (Lucas-Kanade) - Sparse
     * Computes the optical flow for a sparse feature set between two images.
     * Uses OpenMP for parallel point processing.
     */
    void lucas_kanade_flow(
        const float* img1, const float* img2, 
        const float* pts_x, const float* pts_y,
        float* out_u, float* out_v,
        size_t num_pts, size_t H, size_t W, 
        int window_size = 15
    );

    /**
     * Dense Optical Flow (Farneback Approximation)
     * Extremely fast dense flow estimation using polynomial expansion.
     * Output is a 2-channel tensor (U, V) of shape [2, H, W].
     */
    void farneback_flow(
        const float* img1, const float* img2,
        float* flow_out, // [2, H, W]
        size_t H, size_t W,
        int window_size = 15, int iterations = 3
    );

    /**
     * Eulerian Video Magnification (Temporal Bandpass)
     * Amplifies micro-motions (like a heartbeat or vibration) in a spatial map.
     */
    void motion_amplify(
        const float* current_frame, const float* background_avg,
        float* out_frame,
        size_t H, size_t W,
        float amplification_factor = 10.0f
    );

    /**
     * Adaptive Background Subtraction
     * Maintains a running temporal average to extract moving foreground objects.
     */
    void background_subtract(
        const float* current_frame, float* background_model, float* fg_mask,
        size_t H, size_t W,
        float learning_rate = 0.05f, float threshold = 25.0f
    );

} // namespace motion
} // namespace vision
} // namespace mecan
