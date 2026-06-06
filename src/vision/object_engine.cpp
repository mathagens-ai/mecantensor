// HOG, LBP, Connected Components, Contour Extraction, Edge Density,
// Shape Complexity, Texture Energy, Bilateral Filter, Morphology

#include <cstddef>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>
#include <omp.h>

namespace mecan {
namespace vision {
namespace object {

    // ─── 1. HOG (Histogram of Oriented Gradients) ───────────────────────────
    // Computes HOG descriptor for detection/classification
    // cell_size: typically 8, num_bins: typically 9
    void hog_descriptor(
        const float* gray, size_t H, size_t W,
        float* hog_out, int cell_size, int num_bins
    ) {
        int cells_y = (int)H / cell_size;
        int cells_x = (int)W / cell_size;
        int total_bins = cells_y * cells_x * num_bins;
        std::memset(hog_out, 0, total_bins * sizeof(float));

        #pragma omp parallel for
        for (int cy = 0; cy < cells_y; ++cy) {
            for (int cx = 0; cx < cells_x; ++cx) {
                float* cell_hist = hog_out + (cy * cells_x + cx) * num_bins;
                for (int dy = 0; dy < cell_size; ++dy) {
                    for (int dx = 0; dx < cell_size; ++dx) {
                        int y = cy * cell_size + dy;
                        int x = cx * cell_size + dx;
                        if (y < 1 || y >= H-1 || x < 1 || x >= W-1) continue;
                        float gx = gray[y*W+(x+1)] - gray[y*W+(x-1)];
                        float gy = gray[(y+1)*W+x] - gray[(y-1)*W+x];
                        float mag = std::sqrt(gx*gx + gy*gy);
                        float angle = std::atan2(gy, gx) * 180.0f / 3.14159265f;
                        if (angle < 0) angle += 180.0f;
                        int bin = std::min(num_bins - 1, (int)(angle / (180.0f / num_bins)));
                        cell_hist[bin] += mag;
                    }
                }
            }
        }
    }

    // ─── 2. LBP (Local Binary Pattern) — Texture Descriptor ────────────────
    void lbp(const float* gray, float* lbp_out, size_t H, size_t W) {
        std::memset(lbp_out, 0, H * W * sizeof(float));
        #pragma omp parallel for
        for (int y = 1; y < (int)H - 1; ++y) {
            for (int x = 1; x < (int)W - 1; ++x) {
                float center = gray[y*W+x];
                unsigned char code = 0;
                code |= (gray[(y-1)*W+(x-1)] >= center) << 7;
                code |= (gray[(y-1)*W+x]     >= center) << 6;
                code |= (gray[(y-1)*W+(x+1)] >= center) << 5;
                code |= (gray[y*W+(x+1)]     >= center) << 4;
                code |= (gray[(y+1)*W+(x+1)] >= center) << 3;
                code |= (gray[(y+1)*W+x]     >= center) << 2;
                code |= (gray[(y+1)*W+(x-1)] >= center) << 1;
                code |= (gray[y*W+(x-1)]     >= center) << 0;
                lbp_out[y*W+x] = (float)code;
            }
        }
    }

    // ─── 3. Connected Component Labeling (4-connected) ──────────────────────
    int connected_components(
        const float* binary, int* labels, size_t H, size_t W
    ) {
        std::memset(labels, 0, H * W * sizeof(int));
        int current_label = 0;

        for (int y = 0; y < (int)H; ++y) {
            for (int x = 0; x < (int)W; ++x) {
                if (binary[y*W+x] > 0.5f && labels[y*W+x] == 0) {
                    current_label++;
                    // BFS flood fill
                    std::vector<int> stack;
                    stack.push_back(y * W + x);
                    labels[y*W+x] = current_label;
                    while (!stack.empty()) {
                        int idx = stack.back(); stack.pop_back();
                        int py = idx / W, px = idx % W;
                        int neighbors[4][2] = {{py-1,px},{py+1,px},{py,px-1},{py,px+1}};
                        for (auto& n : neighbors) {
                            int ny=n[0], nx=n[1];
                            if (ny>=0 && ny<(int)H && nx>=0 && nx<(int)W &&
                                binary[ny*W+nx]>0.5f && labels[ny*W+nx]==0) {
                                labels[ny*W+nx] = current_label;
                                stack.push_back(ny*W+nx);
                            }
                        }
                    }
                }
            }
        }
        return current_label;
    }

    // ─── 4. Edge Density (fraction of edge pixels) ──────────────────────────
    float edge_density(const float* edge_map, size_t H, size_t W) {
        int count = 0;
        #pragma omp parallel for reduction(+:count)
        for (int i = 0; i < (int)(H * W); ++i) {
            if (edge_map[i] > 0.5f) count++;
        }
        return (float)count / (float)(H * W);
    }

    // ─── 5. Corner Count (from response map) ───────────────────────────────
    int corner_count(const float* response, size_t H, size_t W, float threshold) {
        int count = 0;
        #pragma omp parallel for reduction(+:count)
        for (int i = 0; i < (int)(H * W); ++i) {
            if (response[i] > threshold) count++;
        }
        return count;
    }

    // ─── 6. Shape Complexity Score ──────────────────────────────────────────
    // Based on edge density + corner density + gradient energy
    float shape_complexity(const float* gray, size_t H, size_t W) {
        float grad_energy = 0;
        int edge_count = 0;
        int corner_est = 0;

        #pragma omp parallel for reduction(+:grad_energy, edge_count, corner_est)
        for (int y = 1; y < (int)H - 1; ++y) {
            for (int x = 1; x < (int)W - 1; ++x) {
                float gx = gray[y*W+(x+1)] - gray[y*W+(x-1)];
                float gy = gray[(y+1)*W+x] - gray[(y-1)*W+x];
                float mag = std::sqrt(gx*gx + gy*gy);
                grad_energy += mag;
                if (mag > 0.1f) edge_count++;
                // Quick corner estimate: both gradients strong
                if (std::abs(gx) > 0.08f && std::abs(gy) > 0.08f) corner_est++;
            }
        }
        float n = (float)((H-2)*(W-2));
        return (grad_energy/n)*0.4f + (edge_count/n)*0.3f + (corner_est/n)*0.3f;
    }

    // ─── 7. Texture Energy (Laplacian variance) ────────────────────────────
    float texture_energy(const float* gray, size_t H, size_t W) {
        float sum = 0, sum2 = 0;
        int count = 0;
        #pragma omp parallel for reduction(+:sum, sum2, count)
        for (int y = 1; y < (int)H - 1; ++y) {
            for (int x = 1; x < (int)W - 1; ++x) {
                float lap = -4.0f*gray[y*W+x]
                    + gray[(y-1)*W+x] + gray[(y+1)*W+x]
                    + gray[y*W+(x-1)] + gray[y*W+(x+1)];
                sum += lap;
                sum2 += lap * lap;
                count++;
            }
        }
        float mean = sum / count;
        return sum2 / count - mean * mean; // variance
    }

    // ─── 8. Bilateral Filter (edge-preserving smooth) ──────────────────────
    void bilateral_filter(
        const float* in, float* out, size_t H, size_t W,
        int radius, float sigma_space, float sigma_color
    ) {
        float inv_2ss = -1.0f / (2.0f * sigma_space * sigma_space);
        float inv_2sc = -1.0f / (2.0f * sigma_color * sigma_color);

        #pragma omp parallel for
        for (int y = 0; y < (int)H; ++y) {
            for (int x = 0; x < (int)W; ++x) {
                float center = in[y*W+x];
                float sum_w = 0, sum_v = 0;
                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        int ny = std::max(0, std::min((int)H-1, y+dy));
                        int nx = std::max(0, std::min((int)W-1, x+dx));
                        float val = in[ny*W+nx];
                        float ds = (float)(dx*dx + dy*dy);
                        float dc = (val - center) * (val - center);
                        float w = std::exp(ds * inv_2ss + dc * inv_2sc);
                        sum_w += w;
                        sum_v += val * w;
                    }
                }
                out[y*W+x] = sum_v / sum_w;
            }
        }
    }

    // ─── 9. Morphological Erode ─────────────────────────────────────────────
    void morph_erode(const float* in, float* out, size_t H, size_t W, int radius) {
        #pragma omp parallel for
        for (int y = 0; y < (int)H; ++y) {
            for (int x = 0; x < (int)W; ++x) {
                float mn = 1e9f;
                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        int ny = std::max(0, std::min((int)H-1, y+dy));
                        int nx = std::max(0, std::min((int)W-1, x+dx));
                        mn = std::min(mn, in[ny*W+nx]);
                    }
                }
                out[y*W+x] = mn;
            }
        }
    }

    // ─── 10. Morphological Dilate ───────────────────────────────────────────
    void morph_dilate(const float* in, float* out, size_t H, size_t W, int radius) {
        #pragma omp parallel for
        for (int y = 0; y < (int)H; ++y) {
            for (int x = 0; x < (int)W; ++x) {
                float mx = -1e9f;
                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        int ny = std::max(0, std::min((int)H-1, y+dy));
                        int nx = std::max(0, std::min((int)W-1, x+dx));
                        mx = std::max(mx, in[ny*W+nx]);
                    }
                }
                out[y*W+x] = mx;
            }
        }
    }

    // ─── 11. Adaptive Threshold ─────────────────────────────────────────────
    void adaptive_threshold(
        const float* gray, float* out, size_t H, size_t W,
        int block_size, float C
    ) {
        int half = block_size / 2;
        #pragma omp parallel for
        for (int y = 0; y < (int)H; ++y) {
            for (int x = 0; x < (int)W; ++x) {
                float sum = 0; int count = 0;
                for (int dy = -half; dy <= half; ++dy) {
                    for (int dx = -half; dx <= half; ++dx) {
                        int ny = std::max(0, std::min((int)H-1, y+dy));
                        int nx = std::max(0, std::min((int)W-1, x+dx));
                        sum += gray[ny*W+nx]; count++;
                    }
                }
                float local_mean = sum / count;
                out[y*W+x] = (gray[y*W+x] > local_mean - C) ? 1.0f : 0.0f;
            }
        }
    }

    // ─── 12. Image Gradient Magnitude + Direction ───────────────────────────
    void gradient_map(
        const float* gray, float* magnitude, float* direction,
        size_t H, size_t W
    ) {
        #pragma omp parallel for
        for (int y = 1; y < (int)H - 1; ++y) {
            for (int x = 1; x < (int)W - 1; ++x) {
                float gx = gray[y*W+(x+1)] - gray[y*W+(x-1)];
                float gy = gray[(y+1)*W+x] - gray[(y-1)*W+x];
                magnitude[y*W+x] = std::sqrt(gx*gx + gy*gy);
                direction[y*W+x] = std::atan2(gy, gx) * 180.0f / 3.14159265f;
            }
        }
    }

} // namespace object
} // namespace vision
} // namespace mecan
