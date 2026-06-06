#include "mecan/vision/vector.h"
#include <cmath>
#include <algorithm>
#include <vector>

namespace mecan {
namespace vision {
namespace vector {

    size_t extract_vertices(
        const float* edge_map, size_t H, size_t W,
        Point2D* out_vertices, size_t max_vertices
    ) {
        size_t count = 0;
        for (int y = 1; y < (int)H - 1; ++y) {
            for (int x = 1; x < (int)W - 1; ++x) {
                if (edge_map[y * W + x] > 0.5f) {
                    // Simple Harris approximation
                    float Ix = edge_map[y * W + x + 1] - edge_map[y * W + x - 1];
                    float Iy = edge_map[(y + 1) * W + x] - edge_map[(y - 1) * W + x];
                    if (std::abs(Ix) > 0.5f && std::abs(Iy) > 0.5f) {
                        #pragma omp critical
                        {
                            if (count < max_vertices) {
                                out_vertices[count] = {(float)x, (float)y};
                                count++;
                            }
                        }
                    }
                }
            }
        }
        return count;
    }

    size_t fit_lines(
        const float* edge_map, size_t H, size_t W,
        LineSegment* out_segments, size_t max_segments,
        float tolerance
    ) {
        // Mock simple line fitting for Core
        size_t count = 0;
        for (int y = 0; y < (int)H && count < max_segments; y += 10) {
            out_segments[count] = {{0.0f, (float)y}, {(float)W, (float)y}};
            count++;
        }
        return count;
    }

    size_t polygonize(
        const Point2D* pts, size_t num_pts,
        Point2D* out_pts, size_t max_out,
        float epsilon
    ) {
        if (num_pts == 0 || max_out == 0) return 0;
        out_pts[0] = pts[0];
        if (num_pts > 1 && max_out > 1) {
            out_pts[1] = pts[num_pts - 1];
            return 2;
        }
        return 1;
    }

    void signed_distance_field(
        const float* binary_mask, float* out_sdf,
        size_t H, size_t W
    ) {
        const float INF = 1e6f;
        std::fill(out_sdf, out_sdf + H * W, INF);
        
        #pragma omp parallel for
        for (int i = 0; i < (int)(H * W); ++i) {
            if (binary_mask[i] > 0.5f) {
                out_sdf[i] = 0.0f;
            }
        }

        // Fast Sweeping Method (4 passes)
        for (int y = 0; y < (int)H; ++y) {
            for (int x = 0; x < (int)W; ++x) {
                float v = out_sdf[y * W + x];
                if (x > 0) v = std::min(v, out_sdf[y * W + x - 1] + 1.0f);
                if (y > 0) v = std::min(v, out_sdf[(y - 1) * W + x] + 1.0f);
                out_sdf[y * W + x] = v;
            }
        }
        for (int y = (int)H - 1; y >= 0; --y) {
            for (int x = (int)W - 1; x >= 0; --x) {
                float v = out_sdf[y * W + x];
                if (x < W - 1) v = std::min(v, out_sdf[y * W + x + 1] + 1.0f);
                if (y < H - 1) v = std::min(v, out_sdf[(y + 1) * W + x] + 1.0f);
                out_sdf[y * W + x] = v;
            }
        }
        for (int y = (int)H - 1; y >= 0; --y) {
            for (int x = 0; x < (int)W; ++x) {
                float v = out_sdf[y * W + x];
                if (x > 0) v = std::min(v, out_sdf[y * W + x - 1] + 1.0f);
                if (y < H - 1) v = std::min(v, out_sdf[(y + 1) * W + x] + 1.0f);
                out_sdf[y * W + x] = v;
            }
        }
        for (int y = 0; y < (int)H; ++y) {
            for (int x = (int)W - 1; x >= 0; --x) {
                float v = out_sdf[y * W + x];
                if (x < W - 1) v = std::min(v, out_sdf[y * W + x + 1] + 1.0f);
                if (y > 0) v = std::min(v, out_sdf[(y - 1) * W + x] + 1.0f);
                out_sdf[y * W + x] = v;
            }
        }
    }

} // namespace vector
} // namespace vision
} // namespace mecan
