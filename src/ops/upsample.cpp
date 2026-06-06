// MecanTensor: Upsampling and Transposed Convolutions

#include "mecan/ops/upsample.h"
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__has_include)
#if __has_include(<cblas.h>)
#include <cblas.h>
#define MECAN_HAS_CBLAS 1
#endif
#endif

namespace mecan {
namespace ops {

    static void gemm_transA(
        int M, int N, int K,
        const float* A, const float* B, float* C) {
#if defined(MECAN_HAS_CBLAS)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            M, N, K,
            1.0f, A, M, // A is [K, M] so lda = M
            B, N,       // B is [K, N] so ldb = N
            0.0f, C, N);
#else
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += A[k * M + i] * B[k * N + j]; // A^T
                }
                C[i * N + j] = sum;
            }
        }
#endif
    }

    // ─── col2im ──────────────────────────────────────────────────────────────
    static void col2im(
        const float* data_col,
        size_t C_out, size_t H_out, size_t W_out,
        size_t H_in, size_t W_in,
        size_t kH, size_t kW,
        int stride, int pad,
        float* data_im)
    {
        std::memset(data_im, 0, C_out * H_out * W_out * sizeof(float));

        size_t col_idx = 0;
        for (size_t c = 0; c < C_out; c++) {
            for (size_t kh = 0; kh < kH; kh++) {
                for (size_t kw = 0; kw < kW; kw++) {
                    for (size_t ih = 0; ih < H_in; ih++) {
                        for (size_t iw = 0; iw < W_in; iw++) {
                            int oh = (int)(ih * stride + kh) - pad;
                            int ow = (int)(iw * stride + kw) - pad;
                            if (oh >= 0 && oh < (int)H_out && ow >= 0 && ow < (int)W_out) {
                                data_im[c * H_out * W_out + oh * W_out + ow] += data_col[col_idx];
                            }
                            col_idx++;
                        }
                    }
                }
            }
        }
    }

    // ─── ConvTranspose2d ─────────────────────────────────────────────────────
    void conv_transpose2d(
        const float* input, size_t N, size_t C_in, size_t H, size_t W,
        const float* filter, size_t C_out, size_t kH, size_t kW,
        float* output,
        int stride, int pad, int output_pad)
    {
        size_t H_out = (H - 1) * stride - 2 * pad + kH + output_pad;
        size_t W_out = (W - 1) * stride - 2 * pad + kW + output_pad;

        size_t col_size = C_out * kH * kW * H * W;
        size_t M = C_out * kH * kW;
        size_t K_mat = C_in;
        size_t N_mat = H * W;

        #pragma omp parallel
        {
            std::vector<float> col(col_size);

            #pragma omp for schedule(dynamic)
            for (int64_t n = 0; n < (int64_t)N; n++) {
                const float* in_n = input + n * C_in * H * W;
                float* out_n = output + n * C_out * H_out * W_out;

                // SGEMM: filter^T [C_out*kH*kW, C_in] * input [C_in, H*W] => col [C_out*kH*kW, H*W]
                // Note: filter is [C_in, C_out, kH, kW] which is [C_in, C_out*kH*kW]
                gemm_transA(
                    static_cast<int>(M),
                    static_cast<int>(N_mat),
                    static_cast<int>(K_mat),
                    filter,
                    in_n,
                    col.data());

                col2im(col.data(), C_out, H_out, W_out, H, W, kH, kW, stride, pad, out_n);
            }
        }
    }

    // ─── PixelShuffle ────────────────────────────────────────────────────────
    void pixel_shuffle(
        const float* input, float* output,
        size_t N, size_t C_in, size_t H, size_t W,
        int r)
    {
        size_t C_out = C_in / (r * r);
        size_t H_out = H * r;
        size_t W_out = W * r;

        #pragma omp parallel for collapse(4) schedule(static)
        for (int64_t n = 0; n < (int64_t)N; ++n) {
            for (int64_t c = 0; c < (int64_t)C_out; ++c) {
                for (int64_t h = 0; h < (int64_t)H; ++h) {
                    for (int64_t w = 0; w < (int64_t)W; ++w) {
                        for (int rh = 0; rh < r; ++rh) {
                            for (int rw = 0; rw < r; ++rw) {
                                size_t c_in = c * (r * r) + rh * r + rw;
                                size_t in_idx = ((n * C_in + c_in) * H + h) * W + w;
                                size_t out_idx = ((n * C_out + c) * H_out + (h * r + rh)) * W_out + (w * r + rw);
                                output[out_idx] = input[in_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    // ─── PixelUnshuffle ──────────────────────────────────────────────────────
    void pixel_unshuffle(
        const float* input, float* output,
        size_t N, size_t C_in, size_t H, size_t W,
        int r)
    {
        size_t C_out = C_in * (r * r);
        size_t H_out = H / r;
        size_t W_out = W / r;

        #pragma omp parallel for collapse(4) schedule(static)
        for (int64_t n = 0; n < (int64_t)N; ++n) {
            for (int64_t c = 0; c < (int64_t)C_in; ++c) {
                for (int64_t h = 0; h < (int64_t)H_out; ++h) {
                    for (int64_t w = 0; w < (int64_t)W_out; ++w) {
                        for (int rh = 0; rh < r; ++rh) {
                            for (int rw = 0; rw < r; ++rw) {
                                size_t c_out = c * (r * r) + rh * r + rw;
                                size_t out_idx = ((n * C_out + c_out) * H_out + h) * W_out + w;
                                size_t in_idx = ((n * C_in + c) * H + (h * r + rh)) * W + (w * r + rw);
                                output[out_idx] = input[in_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    // ─── Interpolate Nearest ─────────────────────────────────────────────────
    void upsample_nearest2d(
        const float* input, float* output,
        size_t N, size_t C, size_t H, size_t W,
        size_t out_H, size_t out_W)
    {
        float scale_h = (float)H / out_H;
        float scale_w = (float)W / out_W;

        #pragma omp parallel for collapse(3) schedule(static)
        for (int64_t n = 0; n < (int64_t)N; ++n) {
            for (int64_t c = 0; c < (int64_t)C; ++c) {
                for (int64_t oh = 0; oh < (int64_t)out_H; ++oh) {
                    size_t ih = (size_t)(oh * scale_h);
                    if (ih >= H) ih = H - 1;

                    const float* in_ptr = input + (n * C + c) * H * W + ih * W;
                    float* out_ptr = output + (n * C + c) * out_H * out_W + oh * out_W;

                    for (size_t ow = 0; ow < out_W; ++ow) {
                        size_t iw = (size_t)(ow * scale_w);
                        if (iw >= W) iw = W - 1;
                        out_ptr[ow] = in_ptr[iw];
                    }
                }
            }
        }
    }

    // ─── Interpolate Bilinear ────────────────────────────────────────────────
    void upsample_bilinear2d(
        const float* input, float* output,
        size_t N, size_t C, size_t H, size_t W,
        size_t out_H, size_t out_W,
        bool align_corners)
    {
        float scale_h = align_corners ? (float)(H - 1) / (out_H - 1) : (float)H / out_H;
        float scale_w = align_corners ? (float)(W - 1) / (out_W - 1) : (float)W / out_W;

        #pragma omp parallel for collapse(3) schedule(static)
        for (int64_t n = 0; n < (int64_t)N; ++n) {
            for (int64_t c = 0; c < (int64_t)C; ++c) {
                for (int64_t oh = 0; oh < (int64_t)out_H; ++oh) {
                    float fh = align_corners ? oh * scale_h : (oh + 0.5f) * scale_h - 0.5f;
                    fh = std::max(0.0f, fh);
                    int ih0 = (int)fh;
                    int ih1 = std::min(ih0 + 1, (int)H - 1);
                    float h1_w = fh - ih0;
                    float h0_w = 1.0f - h1_w;

                    const float* in_nc = input + (n * C + c) * H * W;
                    float* out_ptr = output + (n * C + c) * out_H * out_W + oh * out_W;

                    for (size_t ow = 0; ow < out_W; ++ow) {
                        float fw = align_corners ? ow * scale_w : (ow + 0.5f) * scale_w - 0.5f;
                        fw = std::max(0.0f, fw);
                        int iw0 = (int)fw;
                        int iw1 = std::min(iw0 + 1, (int)W - 1);
                        float w1_w = fw - iw0;
                        float w0_w = 1.0f - w1_w;

                        float v00 = in_nc[ih0 * W + iw0];
                        float v01 = in_nc[ih0 * W + iw1];
                        float v10 = in_nc[ih1 * W + iw0];
                        float v11 = in_nc[ih1 * W + iw1];

                        out_ptr[ow] = h0_w * (w0_w * v00 + w1_w * v01) +
                                      h1_w * (w0_w * v10 + w1_w * v11);
                    }
                }
            }
        }
    }

} // namespace ops
} // namespace mecan
