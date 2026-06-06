// MecanTensor Conv2D / Conv3D — im2col + OpenBLAS SGEMM
// The standard convolution algorithm used by PyTorch, TensorFlow, and cuDNN:
//
// Step 1: im2col — unroll each input receptive field into a column
//   For a 3×3 kernel on a 5×5 input with C_in channels:
//   Each output pixel maps to a column of size C_in * kH * kW = C_in * 9
//   Total columns = H_out * W_out
//   Result: col matrix of shape [C_in*kH*kW, H_out*W_out]
//
// Step 2: SGEMM — multiply filter matrix by column matrix
//   Filter reshaped to [C_out, C_in*kH*kW]
//   col matrix is [C_in*kH*kW, H_out*W_out]
//   Output = Filter × col = [C_out, H_out*W_out]
//
// This converts convolution into matrix multiplication, which OpenBLAS
// executes at 330 GFLOPS using hand-tuned AVX2 assembly.

#include "ops/conv.h"
#include <omp.h>
#include <cstring>
#include <vector>
#include <cstdint>
#if defined(__has_include)
#if __has_include(<cblas.h>)
#include <cblas.h>
#define MECAN_HAS_CBLAS 1
#endif
#endif

namespace mecan {
namespace ops {

    static void gemm_rowmajor(
        int M, int N, int K,
        const float* A, const float* B, float* C) {
#if defined(MECAN_HAS_CBLAS)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K,
            1.0f, A, K,
            B, N,
            0.0f, C, N);
#else
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
#endif
    }

    // ─── im2col: Unroll 2D input patches into column matrix ─────────────────
    static void im2col(
        const float* data_im,
        size_t C, size_t H, size_t W,
        size_t kH, size_t kW,
        int stride, int pad,
        float* data_col)
    {
        size_t H_out = (H + 2 * pad - kH) / stride + 1;
        size_t W_out = (W + 2 * pad - kW) / stride + 1;

        size_t col_idx = 0;
        for (size_t c = 0; c < C; c++) {
            for (size_t kh = 0; kh < kH; kh++) {
                for (size_t kw = 0; kw < kW; kw++) {
                    for (size_t oh = 0; oh < H_out; oh++) {
                        for (size_t ow = 0; ow < W_out; ow++) {
                            int ih = (int)(oh * stride + kh) - pad;
                            int iw = (int)(ow * stride + kw) - pad;
                            if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                                data_col[col_idx] = data_im[c * H * W + ih * W + iw];
                            } else {
                                data_col[col_idx] = 0.0f; // zero-padding
                            }
                            col_idx++;
                        }
                    }
                }
            }
        }
    }

    // ─── Conv2D: im2col + SGEMM ────────────────────────────────────────────
    void conv2d(
        const float* input, size_t N, size_t C_in, size_t H, size_t W,
        const float* filter, size_t C_out, size_t kH, size_t kW,
        float* output,
        int stride, int pad)
    {
        size_t H_out = (H + 2 * pad - kH) / stride + 1;
        size_t W_out = (W + 2 * pad - kW) / stride + 1;

        size_t col_size = C_in * kH * kW * H_out * W_out;
        size_t M = C_out;
        size_t K_mat = C_in * kH * kW;
        size_t N_mat = H_out * W_out;

        // Per-batch processing
        #pragma omp parallel
        {
            // Each thread gets its own col buffer
            std::vector<float> col(col_size);

            #pragma omp for schedule(dynamic)
            for (int64_t n = 0; n < (int64_t)N; n++) {
                const float* in_n = input + n * C_in * H * W;
                float* out_n = output + n * C_out * H_out * W_out;

                // Step 1: im2col
                im2col(in_n, C_in, H, W, kH, kW, stride, pad, col.data());

                // Step 2: SGEMM — filter[C_out, C_in*kH*kW] × col[C_in*kH*kW, H_out*W_out]
                // = output[C_out, H_out*W_out]
                gemm_rowmajor(
                    static_cast<int>(M),
                    static_cast<int>(N_mat),
                    static_cast<int>(K_mat),
                    filter,
                    col.data(),
                    out_n);
            }
        }
    }

    // ─── vol2col: Unroll 3D input patches into column matrix ────────────────
    static void vol2col(
        const float* data_vol,
        size_t C, size_t D, size_t H, size_t W,
        size_t kD, size_t kH, size_t kW,
        int stride, int pad,
        float* data_col)
    {
        size_t D_out = (D + 2 * pad - kD) / stride + 1;
        size_t H_out = (H + 2 * pad - kH) / stride + 1;
        size_t W_out = (W + 2 * pad - kW) / stride + 1;

        size_t col_idx = 0;
        for (size_t c = 0; c < C; c++) {
            for (size_t kd = 0; kd < kD; kd++) {
                for (size_t kh = 0; kh < kH; kh++) {
                    for (size_t kw = 0; kw < kW; kw++) {
                        for (size_t od = 0; od < D_out; od++) {
                            for (size_t oh = 0; oh < H_out; oh++) {
                                for (size_t ow = 0; ow < W_out; ow++) {
                                    int id = (int)(od * stride + kd) - pad;
                                    int ih = (int)(oh * stride + kh) - pad;
                                    int iw = (int)(ow * stride + kw) - pad;
                                    if (id >= 0 && id < (int)D &&
                                        ih >= 0 && ih < (int)H &&
                                        iw >= 0 && iw < (int)W) {
                                        data_col[col_idx] = data_vol[c*D*H*W + id*H*W + ih*W + iw];
                                    } else {
                                        data_col[col_idx] = 0.0f;
                                    }
                                    col_idx++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ─── Conv3D: vol2col + SGEMM ───────────────────────────────────────────
    void conv3d(
        const float* input, size_t N, size_t C_in, size_t D, size_t H, size_t W,
        const float* filter, size_t C_out, size_t kD, size_t kH, size_t kW,
        float* output,
        int stride, int pad)
    {
        size_t D_out = (D + 2 * pad - kD) / stride + 1;
        size_t H_out = (H + 2 * pad - kH) / stride + 1;
        size_t W_out = (W + 2 * pad - kW) / stride + 1;

        size_t col_size = C_in * kD * kH * kW * D_out * H_out * W_out;
        size_t M = C_out;
        size_t K_mat = C_in * kD * kH * kW;
        size_t N_mat = D_out * H_out * W_out;

        #pragma omp parallel
        {
            std::vector<float> col(col_size);

            #pragma omp for schedule(dynamic)
            for (int64_t n = 0; n < (int64_t)N; n++) {
                const float* in_n = input + n * C_in * D * H * W;
                float* out_n = output + n * C_out * D_out * H_out * W_out;

                vol2col(in_n, C_in, D, H, W, kD, kH, kW, stride, pad, col.data());

                gemm_rowmajor(
                    static_cast<int>(M),
                    static_cast<int>(N_mat),
                    static_cast<int>(K_mat),
                    filter,
                    col.data(),
                    out_n);
            }
        }
    }


    void conv1d(
        const float* input,   size_t N, size_t C_in,  size_t L,
        const float* filter,  size_t C_out, size_t kL,
        float* output,
        int stride, int pad
    ) {
        // Simple direct convolution for 1D
        size_t L_out = (L + 2 * pad - kL) / stride + 1;
        
        #pragma omp parallel for collapse(2)
        for (int n = 0; n < N; ++n) {
            for (int co = 0; co < C_out; ++co) {
                for (int lo = 0; lo < L_out; ++lo) {
                    float sum = 0.0f;
                    for (int ci = 0; ci < C_in; ++ci) {
                        for (int k = 0; k < kL; ++k) {
                            int li = lo * stride - pad + k;
                            if (li >= 0 && li < L) {
                                float x = input[n * C_in * L + ci * L + li];
                                float w = filter[co * C_in * kL + ci * kL + k];
                                sum += x * w;
                            }
                        }
                    }
                    output[n * C_out * L_out + co * L_out + lo] = sum;
                }
            }
        }
    }




    void depthwise_conv2d(
        const float* input,   size_t N, size_t C_in,  size_t H,  size_t W,
        const float* filter,  size_t kH, size_t kW,
        float* output,
        int stride, int pad
    ) {
        size_t out_H = (H + 2 * pad - kH) / stride + 1;
        size_t out_W = (W + 2 * pad - kW) / stride + 1;
        
        #pragma omp parallel for collapse(3)
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C_in; ++c) {
                for (int oh = 0; oh < out_H; ++oh) {
                    for (int ow = 0; ow < out_W; ++ow) {
                        float sum = 0.0f;
                        for (int kh = 0; kh < kH; ++kh) {
                            for (int kw = 0; kw < kW; ++kw) {
                                int ih = oh * stride - pad + kh;
                                int iw = ow * stride - pad + kw;
                                
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    float x = input[n * C_in * H * W + c * H * W + ih * W + iw];
                                    float w = filter[c * kH * kW + kh * kW + kw];
                                    sum += x * w;
                                }
                            }
                        }
                        output[n * C_in * out_H * out_W + c * out_H * out_W + oh * out_W + ow] = sum;
                    }
                }
            }
        }
    }

} // namespace ops
} // namespace mecan
