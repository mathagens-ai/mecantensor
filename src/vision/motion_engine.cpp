#include "mecan/vision/motion.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace mecan {
namespace vision {
namespace motion {

    void lucas_kanade_flow(
        const float* img1, const float* img2, 
        const float* pts_x, const float* pts_y,
        float* out_u, float* out_v,
        size_t num_pts, size_t H, size_t W, 
        int window_size
    ) {
        int half_w = window_size / 2;
        #pragma omp parallel for
        for (int i = 0; i < (int)num_pts; ++i) {
            int px = (int)pts_x[i];
            int py = (int)pts_y[i];
            
            float Ixx = 0, Ixy = 0, Iyy = 0;
            float IxIt = 0, IyIt = 0;

            for (int wy = -half_w; wy <= half_w; ++wy) {
                for (int wx = -half_w; wx <= half_w; ++wx) {
                    int y = py + wy;
                    int x = px + wx;
                    
                    if (y > 0 && y < H - 1 && x > 0 && x < W - 1) {
                        float Ix = (img1[y * W + (x + 1)] - img1[y * W + (x - 1)]) * 0.5f;
                        float Iy = (img1[(y + 1) * W + x] - img1[(y - 1) * W + x]) * 0.5f;
                        float It = img2[y * W + x] - img1[y * W + x];

                        Ixx += Ix * Ix;
                        Ixy += Ix * Iy;
                        Iyy += Iy * Iy;
                        IxIt += Ix * It;
                        IyIt += Iy * It;
                    }
                }
            }
            
            float det = Ixx * Iyy - Ixy * Ixy;
            if (det > 1e-5f) {
                out_u[i] = (Ixy * IyIt - Iyy * IxIt) / det;
                out_v[i] = (Ixy * IxIt - Ixx * IyIt) / det;
            } else {
                out_u[i] = 0;
                out_v[i] = 0;
            }
        }
    }

    void farneback_flow(
        const float* img1, const float* img2,
        float* flow_out,
        size_t H, size_t W,
        int window_size, int iterations
    ) {
        int half_w = window_size / 2;
        #pragma omp parallel for
        for (int y = 0; y < (int)H; ++y) {
            for (int x = 0; x < (int)W; ++x) {
                float u = 0.0f;
                float v = 0.0f;
                for (int iter = 0; iter < iterations; ++iter) {
                    float Ixx = 0, Ixy = 0, Iyy = 0;
                    float IxIt = 0, IyIt = 0;
                    
                    for (int wy = -half_w; wy <= half_w; ++wy) {
                        for (int wx = -half_w; wx <= half_w; ++wx) {
                            int cy = std::max(1, std::min((int)H - 2, y + wy));
                            int cx = std::max(1, std::min((int)W - 2, x + wx));
                            int ncx = std::max(0, std::min((int)W - 1, cx + (int)u));
                            int ncy = std::max(0, std::min((int)H - 1, cy + (int)v));

                            float Ix = (img1[cy * W + (cx + 1)] - img1[cy * W + (cx - 1)]) * 0.5f;
                            float Iy = (img1[(cy + 1) * W + cx] - img1[(cy - 1) * W + cx]) * 0.5f;
                            float It = img2[ncy * W + ncx] - img1[cy * W + cx];

                            Ixx += Ix * Ix;
                            Ixy += Ix * Iy;
                            Iyy += Iy * Iy;
                            IxIt += Ix * It;
                            IyIt += Iy * It;
                        }
                    }
                    float det = Ixx * Iyy - Ixy * Ixy;
                    if (det > 1e-5f) {
                        u += (Ixy * IyIt - Iyy * IxIt) / det;
                        v += (Ixy * IxIt - Ixx * IyIt) / det;
                    }
                }
                flow_out[0 * H * W + y * W + x] = u;
                flow_out[1 * H * W + y * W + x] = v;
            }
        }
    }

    void motion_amplify(
        const float* current_frame, const float* background_avg,
        float* out_frame,
        size_t H, size_t W,
        float amplification_factor
    ) {
        #pragma omp parallel for
        for (int i = 0; i < (int)(H * W); ++i) {
            float diff = current_frame[i] - background_avg[i];
            out_frame[i] = current_frame[i] + diff * amplification_factor;
        }
    }

    void background_subtract(
        const float* current_frame, float* background_model, float* fg_mask,
        size_t H, size_t W,
        float learning_rate, float threshold
    ) {
        #pragma omp parallel for
        for (int i = 0; i < (int)(H * W); ++i) {
            float diff = current_frame[i] - background_model[i];
            if (std::abs(diff) > threshold) {
                fg_mask[i] = 1.0f;
            } else {
                fg_mask[i] = 0.0f;
            }
            background_model[i] = (1.0f - learning_rate) * background_model[i] + learning_rate * current_frame[i];
        }
    }

} // namespace motion

namespace temporal {

    // 1. Scene Change Detection (returns change score 0-1)
    float scene_change_score(
        const float* frame1, const float* frame2,
        size_t H, size_t W
    ) {
        float diff_sum = 0;
        #pragma omp parallel for reduction(+:diff_sum)
        for (int i = 0; i < (int)(H * W); ++i) {
            float d = frame2[i] - frame1[i];
            diff_sum += d * d;
        }
        return diff_sum / (float)(H * W);
    }

    // 2. Motion Heatmap Accumulator
    void motion_heatmap(
        const float* frame1, const float* frame2,
        float* heatmap, size_t H, size_t W,
        float decay
    ) {
        #pragma omp parallel for
        for (int i = 0; i < (int)(H * W); ++i) {
            float diff = std::abs(frame2[i] - frame1[i]);
            heatmap[i] = heatmap[i] * decay + diff;
        }
    }

    // 3. Velocity Field Statistics
    void flow_statistics(
        const float* flow, size_t H, size_t W,
        float* avg_mag, float* avg_dir, float* max_mag
    ) {
        float sum_mag = 0, sum_dx = 0, sum_dy = 0, mx = 0;
        #pragma omp parallel for reduction(+:sum_mag, sum_dx, sum_dy)
        for (int i = 0; i < (int)(H * W); ++i) {
            float u = flow[i];
            float v = flow[H * W + i];
            float mag = std::sqrt(u*u + v*v);
            sum_mag += mag;
            sum_dx += u;
            sum_dy += v;
            #pragma omp critical
            { if (mag > mx) mx = mag; }
        }
        float n = (float)(H * W);
        *avg_mag = sum_mag / n;
        *avg_dir = std::atan2(sum_dy / n, sum_dx / n) * 180.0f / 3.14159265f;
        *max_mag = mx;
    }

    // 4. Frame Stability Score (inverse of motion energy)
    float frame_stability(
        const float* frame1, const float* frame2,
        size_t H, size_t W
    ) {
        float energy = 0;
        #pragma omp parallel for reduction(+:energy)
        for (int i = 0; i < (int)(H * W); ++i) {
            float d = frame2[i] - frame1[i];
            energy += d * d;
        }
        float mse = energy / (float)(H * W);
        return 1.0f / (1.0f + mse * 100.0f); // 1.0 = perfectly stable
    }

    // 5. Temporal Difference Accumulator (multi-frame)
    void temporal_diff_accumulate(
        const float* frames, size_t num_frames,
        float* accumulated, size_t H, size_t W
    ) {
        size_t frame_size = H * W;
        std::memset(accumulated, 0, frame_size * sizeof(float));
        for (size_t f = 1; f < num_frames; ++f) {
            #pragma omp parallel for
            for (int i = 0; i < (int)frame_size; ++i) {
                float d = std::abs(frames[f * frame_size + i] - frames[(f-1) * frame_size + i]);
                accumulated[i] += d;
            }
        }
        float norm = 1.0f / (float)(num_frames - 1);
        #pragma omp parallel for
        for (int i = 0; i < (int)frame_size; ++i)
            accumulated[i] *= norm;
    }

    // 6. Flicker Detection (temporal variance)
    float flicker_score(
        const float* frames, size_t num_frames,
        size_t H, size_t W
    ) {
        size_t frame_size = H * W;
        float total_var = 0;
        #pragma omp parallel for reduction(+:total_var)
        for (int i = 0; i < (int)frame_size; ++i) {
            float sum = 0, sum2 = 0;
            for (size_t f = 0; f < num_frames; ++f) {
                float v = frames[f * frame_size + i];
                sum += v; sum2 += v * v;
            }
            float mean = sum / num_frames;
            total_var += sum2 / num_frames - mean * mean;
        }
        return total_var / (float)frame_size;
    }

} // namespace temporal

} // namespace vision
} // namespace mecan

