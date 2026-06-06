// MecanTensor: Detection Utilities — NMS + ROI Align

#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace mecan {
namespace ops {

    // ─── IoU (Intersection over Union) ───────────────────────────────────────

    static float box_iou_single(const float* a, const float* b) {
        // boxes in [x1, y1, x2, y2] format
        float x1 = std::max(a[0], b[0]);
        float y1 = std::max(a[1], b[1]);
        float x2 = std::min(a[2], b[2]);
        float y2 = std::min(a[3], b[3]);
        float inter = std::max(0.f, x2-x1) * std::max(0.f, y2-y1);
        float area_a = (a[2]-a[0]) * (a[3]-a[1]);
        float area_b = (b[2]-b[0]) * (b[3]-b[1]);
        float uni = area_a + area_b - inter;
        return (uni > 0) ? inter / uni : 0.0f;
    }

    // ─── NMS (Non-Maximum Suppression) ───────────────────────────────────────
    // boxes: Nx4 [x1,y1,x2,y2], scores: N
    // Returns indices of kept boxes

    std::vector<int> nms(
        const float* boxes, const float* scores, int N,
        float iou_threshold)
    {
        // Sort by score (descending)
        std::vector<int> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
            [&](int a, int b) { return scores[a] > scores[b]; });

        std::vector<bool> suppressed(N, false);
        std::vector<int> keep;

        for (int i = 0; i < N; ++i) {
            int idx = indices[i];
            if (suppressed[idx]) continue;
            keep.push_back(idx);

            const float* box_i = boxes + idx * 4;
            for (int j = i + 1; j < N; ++j) {
                int jdx = indices[j];
                if (suppressed[jdx]) continue;
                if (box_iou_single(box_i, boxes + jdx * 4) > iou_threshold) {
                    suppressed[jdx] = true;
                }
            }
        }
        return keep;
    }

    // ─── IoU Matrix ──────────────────────────────────────────────────────────

    void iou_matrix(
        const float* boxes_a, int Na,
        const float* boxes_b, int Nb,
        float* out)
    {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t i = 0; i < Na; ++i) {
            for (int64_t j = 0; j < Nb; ++j) {
                out[i * Nb + j] = box_iou_single(boxes_a + i*4, boxes_b + j*4);
            }
        }
    }

    // ─── ROI Align ───────────────────────────────────────────────────────────
    // Bilinear-interpolated feature sampling from region proposals

    void roi_align(
        const float* features, size_t C, size_t H, size_t W,
        const float* rois, int num_rois,  // rois: [batch_idx, x1, y1, x2, y2]
        float* output, int out_H, int out_W,
        float spatial_scale, int sampling_ratio)
    {
        if (sampling_ratio <= 0) sampling_ratio = 2;

        #pragma omp parallel for schedule(dynamic)
        for (int64_t r = 0; r < num_rois; ++r) {
            float x1 = rois[r*5+1] * spatial_scale;
            float y1 = rois[r*5+2] * spatial_scale;
            float x2 = rois[r*5+3] * spatial_scale;
            float y2 = rois[r*5+4] * spatial_scale;

            float bin_h = (y2 - y1) / out_H;
            float bin_w = (x2 - x1) / out_W;

            for (size_t c = 0; c < C; ++c) {
                const float* feat_c = features + c * H * W;
                for (int oh = 0; oh < out_H; ++oh) {
                    for (int ow = 0; ow < out_W; ++ow) {
                        float sum = 0;
                        int count = 0;
                        for (int sy = 0; sy < sampling_ratio; ++sy) {
                            for (int sx = 0; sx < sampling_ratio; ++sx) {
                                float fy = y1 + bin_h * (oh + (sy+0.5f)/sampling_ratio);
                                float fx = x1 + bin_w * (ow + (sx+0.5f)/sampling_ratio);

                                if (fy < 0 || fy >= H-1 || fx < 0 || fx >= W-1) continue;

                                int y_lo = (int)fy, x_lo = (int)fx;
                                int y_hi = y_lo+1, x_hi = x_lo+1;
                                float ly = fy - y_lo, lx = fx - x_lo;
                                float hy = 1-ly, hx = 1-lx;

                                sum += hy*hx*feat_c[y_lo*W+x_lo]
                                     + hy*lx*feat_c[y_lo*W+x_hi]
                                     + ly*hx*feat_c[y_hi*W+x_lo]
                                     + ly*lx*feat_c[y_hi*W+x_hi];
                                count++;
                            }
                        }
                        output[(r*C+c)*out_H*out_W + oh*out_W + ow] =
                            (count > 0) ? sum / count : 0.0f;
                    }
                }
            }
        }
    }

} // namespace ops
} // namespace mecan
