// MecanTensor Vision: Edge & Corner Detection Kernels

#include "mecan/vision/detect.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>
#include <omp.h>

namespace mecan {
namespace vision {
namespace detect {

    // ─── Gaussian blur ──────────────────────────────────────────────────────
    void gaussian_blur(const float* in, float* out, size_t H, size_t W,
                       int kernel_size, float sigma) {
        int half = kernel_size / 2;
        // Build 1D kernel
        std::vector<float> kern(kernel_size);
        float sum = 0;
        for (int i = 0; i < kernel_size; ++i) {
            float x = (float)(i - half);
            kern[i] = std::exp(-x * x / (2.0f * sigma * sigma));
            sum += kern[i];
        }
        for (int i = 0; i < kernel_size; ++i) kern[i] /= sum;

        // Separable: horizontal then vertical
        std::vector<float> tmp(H * W, 0.0f);

        // Horizontal pass
        #pragma omp parallel for
        for (int y = 0; y < (int)H; ++y) {
            for (int x = 0; x < (int)W; ++x) {
                float val = 0;
                for (int k = -half; k <= half; ++k) {
                    int cx = std::max(0, std::min((int)W - 1, x + k));
                    val += in[y * W + cx] * kern[k + half];
                }
                tmp[y * W + x] = val;
            }
        }
        // Vertical pass
        #pragma omp parallel for
        for (int y = 0; y < (int)H; ++y) {
            for (int x = 0; x < (int)W; ++x) {
                float val = 0;
                for (int k = -half; k <= half; ++k) {
                    int cy = std::max(0, std::min((int)H - 1, y + k));
                    val += tmp[cy * W + x] * kern[k + half];
                }
                out[y * W + x] = val;
            }
        }
    }

    // ─── Sobel gradient ─────────────────────────────────────────────────────
    void sobel(const float* gray, float* out, size_t H, size_t W, int axis) {
        #pragma omp parallel for
        for (int y = 1; y < (int)H - 1; ++y) {
            for (int x = 1; x < (int)W - 1; ++x) {
                float gx = -gray[(y-1)*W+(x-1)] + gray[(y-1)*W+(x+1)]
                          -2*gray[y*W+(x-1)]    + 2*gray[y*W+(x+1)]
                          -gray[(y+1)*W+(x-1)]  + gray[(y+1)*W+(x+1)];

                float gy = -gray[(y-1)*W+(x-1)] - 2*gray[(y-1)*W+x] - gray[(y-1)*W+(x+1)]
                          +gray[(y+1)*W+(x-1)]  + 2*gray[(y+1)*W+x] + gray[(y+1)*W+(x+1)];

                if (axis == 0)      out[y*W+x] = gx;
                else if (axis == 1) out[y*W+x] = gy;
                else                out[y*W+x] = std::sqrt(gx*gx + gy*gy);
            }
        }
    }

    // ─── Non-max suppression (edge thinning) ────────────────────────────────
    void non_max_suppression(const float* magnitude, const float* direction,
                             float* out, size_t H, size_t W) {
        std::memset(out, 0, H * W * sizeof(float));
        #pragma omp parallel for
        for (int y = 1; y < (int)H - 1; ++y) {
            for (int x = 1; x < (int)W - 1; ++x) {
                float angle = direction[y*W+x];
                // Quantize to 0, 45, 90, 135 degrees
                if (angle < 0) angle += 180.0f;
                float mag = magnitude[y*W+x];
                float n1, n2;
                if ((angle >= 0 && angle < 22.5f) || (angle >= 157.5f && angle <= 180.0f)) {
                    n1 = magnitude[y*W+(x-1)]; n2 = magnitude[y*W+(x+1)];
                } else if (angle >= 22.5f && angle < 67.5f) {
                    n1 = magnitude[(y-1)*W+(x+1)]; n2 = magnitude[(y+1)*W+(x-1)];
                } else if (angle >= 67.5f && angle < 112.5f) {
                    n1 = magnitude[(y-1)*W+x]; n2 = magnitude[(y+1)*W+x];
                } else {
                    n1 = magnitude[(y-1)*W+(x-1)]; n2 = magnitude[(y+1)*W+(x+1)];
                }
                out[y*W+x] = (mag >= n1 && mag >= n2) ? mag : 0.0f;
            }
        }
    }

    // ─── Hysteresis thresholding ────────────────────────────────────────────
    void hysteresis_threshold(const float* thin_edges, float* out,
                              size_t H, size_t W,
                              float low_thresh, float high_thresh) {
        std::memset(out, 0, H * W * sizeof(float));
        // Mark strong edges
        for (size_t i = 0; i < H * W; ++i) {
            if (thin_edges[i] >= high_thresh) out[i] = 1.0f;
        }
        // Connect weak edges to strong (simple one-pass)
        for (int y = 1; y < (int)H - 1; ++y) {
            for (int x = 1; x < (int)W - 1; ++x) {
                if (thin_edges[y*W+x] >= low_thresh && out[y*W+x] == 0.0f) {
                    // Check 8-neighbours for strong edge
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (out[(y+dy)*W+(x+dx)] == 1.0f) {
                                out[y*W+x] = 1.0f;
                                goto next_pixel;
                            }
                        }
                    }
                    next_pixel:;
                }
            }
        }
    }

    // ─── Canny edge detection ───────────────────────────────────────────────
    void canny(const float* gray, float* edges, size_t H, size_t W,
               float low_threshold, float high_threshold) {
        size_t N = H * W;
        std::vector<float> blurred(N);
        std::vector<float> gx(N, 0.0f), gy(N, 0.0f), mag(N, 0.0f), dir(N, 0.0f);
        std::vector<float> thin(N, 0.0f);

        // Step 1: Gaussian blur
        gaussian_blur(gray, blurred.data(), H, W, 5, 1.4f);

        // Step 2: Sobel gradients
        sobel(blurred.data(), gx.data(), H, W, 0);
        sobel(blurred.data(), gy.data(), H, W, 1);

        // Step 3: Magnitude + direction
        #pragma omp parallel for
        for (int i = 0; i < (int)N; ++i) {
            mag[i] = std::sqrt(gx[i]*gx[i] + gy[i]*gy[i]);
            dir[i] = std::atan2(gy[i], gx[i]) * 180.0f / 3.14159265f;
        }

        // Step 4: Non-max suppression
        non_max_suppression(mag.data(), dir.data(), thin.data(), H, W);

        // Step 5: Hysteresis
        hysteresis_threshold(thin.data(), edges, H, W, low_threshold, high_threshold);
    }

    // ─── Laplacian of Gaussian ──────────────────────────────────────────────
    void laplacian(const float* gray, float* out, size_t H, size_t W) {
        #pragma omp parallel for
        for (int y = 1; y < (int)H - 1; ++y) {
            for (int x = 1; x < (int)W - 1; ++x) {
                out[y*W+x] = -4.0f * gray[y*W+x]
                    + gray[(y-1)*W+x] + gray[(y+1)*W+x]
                    + gray[y*W+(x-1)] + gray[y*W+(x+1)];
            }
        }
    }

    // ─── Harris corner response ─────────────────────────────────────────────
    void harris_corners(const float* gray, float* response, size_t H, size_t W,
                        float k, int block_size) {
        std::vector<float> Ix(H*W, 0.0f), Iy(H*W, 0.0f);
        sobel(gray, Ix.data(), H, W, 0);
        sobel(gray, Iy.data(), H, W, 1);

        int half = block_size / 2;
        #pragma omp parallel for
        for (int y = half; y < (int)H - half; ++y) {
            for (int x = half; x < (int)W - half; ++x) {
                float Sxx = 0, Sxy = 0, Syy = 0;
                for (int wy = -half; wy <= half; ++wy) {
                    for (int wx = -half; wx <= half; ++wx) {
                        float ix = Ix[(y+wy)*W+(x+wx)];
                        float iy = Iy[(y+wy)*W+(x+wx)];
                        Sxx += ix * ix;
                        Sxy += ix * iy;
                        Syy += iy * iy;
                    }
                }
                float det = Sxx * Syy - Sxy * Sxy;
                float trace = Sxx + Syy;
                response[y*W+x] = det - k * trace * trace;
            }
        }
    }

    void shi_tomasi(const float* gray, float* response, size_t H, size_t W,
                    int block_size) {
        std::vector<float> Ix(H*W, 0.0f), Iy(H*W, 0.0f);
        sobel(gray, Ix.data(), H, W, 0);
        sobel(gray, Iy.data(), H, W, 1);

        int half = block_size / 2;
        #pragma omp parallel for
        for (int y = half; y < (int)H - half; ++y) {
            for (int x = half; x < (int)W - half; ++x) {
                float Sxx = 0, Sxy = 0, Syy = 0;
                for (int wy = -half; wy <= half; ++wy) {
                    for (int wx = -half; wx <= half; ++wx) {
                        float ix = Ix[(y+wy)*W+(x+wx)];
                        float iy = Iy[(y+wy)*W+(x+wx)];
                        Sxx += ix * ix;
                        Sxy += ix * iy;
                        Syy += iy * iy;
                    }
                }
                float trace = Sxx + Syy;
                float det = Sxx * Syy - Sxy * Sxy;
                float disc = std::sqrt(std::max(0.0f, trace * trace * 0.25f - det));
                response[y*W+x] = trace * 0.5f - disc; // min eigenvalue
            }
        }
    }

} // namespace detect
} // namespace vision
} // namespace mecan
