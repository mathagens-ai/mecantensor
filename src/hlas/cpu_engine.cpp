// MecanTensor HLAS: Universal CPU Engine
// Implements Cache-Blocked SGEMM with fused activations and architecture 
// dispatch (AVX2, NEON, Fallback).

#include "hlas.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <omp.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace mecan {
namespace hlas {

namespace {

    // SIMD-optimized Micro-kernel (6x16 block for AVX2, or similar for NEON)
    // Computes a 6x16 block of C using A and B in registers.
    inline void micro_kernel_avx2(
        int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc,
        FusedActivation activation) 
    {
#if defined(__AVX2__)
        // We accumulate into 12 AVX2 registers (6 rows x 2 vectors of 8 floats) = 6x16 block
        __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
        __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
        __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
        __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
        __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
        __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

        for (int k = 0; k < K; ++k) {
            __m256 b0 = _mm256_loadu_ps(B + k * ldb + 0);
            __m256 b1 = _mm256_loadu_ps(B + k * ldb + 8);

            __m256 a0 = _mm256_set1_ps(A[0 * lda + k]);
            c00 = _mm256_fmadd_ps(a0, b0, c00); c01 = _mm256_fmadd_ps(a0, b1, c01);

            __m256 a1 = _mm256_set1_ps(A[1 * lda + k]);
            c10 = _mm256_fmadd_ps(a1, b0, c10); c11 = _mm256_fmadd_ps(a1, b1, c11);

            __m256 a2 = _mm256_set1_ps(A[2 * lda + k]);
            c20 = _mm256_fmadd_ps(a2, b0, c20); c21 = _mm256_fmadd_ps(a2, b1, c21);

            __m256 a3 = _mm256_set1_ps(A[3 * lda + k]);
            c30 = _mm256_fmadd_ps(a3, b0, c30); c31 = _mm256_fmadd_ps(a3, b1, c31);

            __m256 a4 = _mm256_set1_ps(A[4 * lda + k]);
            c40 = _mm256_fmadd_ps(a4, b0, c40); c41 = _mm256_fmadd_ps(a4, b1, c41);

            __m256 a5 = _mm256_set1_ps(A[5 * lda + k]);
            c50 = _mm256_fmadd_ps(a5, b0, c50); c51 = _mm256_fmadd_ps(a5, b1, c51);
        }

        // Apply Fused Activations directly in registers BEFORE writing to RAM
        if (activation == FusedActivation::RELU) {
            __m256 z = _mm256_setzero_ps();
            c00 = _mm256_max_ps(c00, z); c01 = _mm256_max_ps(c01, z);
            c10 = _mm256_max_ps(c10, z); c11 = _mm256_max_ps(c11, z);
            c20 = _mm256_max_ps(c20, z); c21 = _mm256_max_ps(c21, z);
            c30 = _mm256_max_ps(c30, z); c31 = _mm256_max_ps(c31, z);
            c40 = _mm256_max_ps(c40, z); c41 = _mm256_max_ps(c41, z);
            c50 = _mm256_max_ps(c50, z); c51 = _mm256_max_ps(c51, z);
        }

        // Store
        _mm256_storeu_ps(C + 0 * ldc + 0, _mm256_add_ps(_mm256_loadu_ps(C + 0 * ldc + 0), c00));
        _mm256_storeu_ps(C + 0 * ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 0 * ldc + 8), c01));
        _mm256_storeu_ps(C + 1 * ldc + 0, _mm256_add_ps(_mm256_loadu_ps(C + 1 * ldc + 0), c10));
        _mm256_storeu_ps(C + 1 * ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 1 * ldc + 8), c11));
        _mm256_storeu_ps(C + 2 * ldc + 0, _mm256_add_ps(_mm256_loadu_ps(C + 2 * ldc + 0), c20));
        _mm256_storeu_ps(C + 2 * ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 2 * ldc + 8), c21));
        _mm256_storeu_ps(C + 3 * ldc + 0, _mm256_add_ps(_mm256_loadu_ps(C + 3 * ldc + 0), c30));
        _mm256_storeu_ps(C + 3 * ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 3 * ldc + 8), c31));
        _mm256_storeu_ps(C + 4 * ldc + 0, _mm256_add_ps(_mm256_loadu_ps(C + 4 * ldc + 0), c40));
        _mm256_storeu_ps(C + 4 * ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 4 * ldc + 8), c41));
        _mm256_storeu_ps(C + 5 * ldc + 0, _mm256_add_ps(_mm256_loadu_ps(C + 5 * ldc + 0), c50));
        _mm256_storeu_ps(C + 5 * ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 5 * ldc + 8), c51));
#endif
    }

    inline void micro_kernel_generic(
        int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc,
        FusedActivation activation)
    {
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                float a = A[i * lda + k];
                for (int j = 0; j < N; ++j) {
                    C[i * ldc + j] += a * B[k * ldb + j];
                }
            }
        }
        
        // Fused Activation generic
        if (activation != FusedActivation::NONE) {
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    float& val = C[i * ldc + j];
                    if (activation == FusedActivation::RELU) {
                        if (val < 0.0f) val = 0.0f;
                    } else if (activation == FusedActivation::SILU) {
                        val = val / (1.0f + std::exp(-val));
                    }
                }
            }
        }
    }
} // anonymous namespace

    CpuEngine::CpuEngine() { calibrate(); }

    void CpuEngine::calibrate() {
        cache_.l1_data_size = 48 * 1024;
        cache_.l2_size = 1280 * 1024;
        cache_.l3_size = 24 * 1024 * 1024;

#if defined(__AVX512F__)
        arch_ = CpuArch::X86_AVX512;
#elif defined(__AVX2__)
        arch_ = CpuArch::X86_AVX2;
#elif defined(__ARM_NEON) || defined(__aarch64__)
        arch_ = CpuArch::ARM_NEON;
#else
        arch_ = CpuArch::GENERIC;
#endif
    }

    DeviceInfo CpuEngine::get_device_info() const {
        return {"Generic CPU", BackendType::CPU, 0, 16};
    }

    void CpuEngine::sgemm(
        int M, int N, int K,
        float alpha,
        const float* A, int lda,
        const float* B, int ldb,
        float beta,
        float* C, int ldc,
        FusedActivation activation)
    {
        // Handle beta scale
        if (beta == 0.0f) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < M; ++i) {
                std::memset(C + i * ldc, 0, N * sizeof(float));
            }
        } else if (beta != 1.0f) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    C[i * ldc + j] *= beta;
                }
            }
        }

        // Cache Blocking Parameters (Tuned for L1/L2 sizes)
        // MC * KC fits in L2, KC * NC fits in L3, MR * NR fits in registers
        int MC = 256; 
        int KC = 256; 
        int NC = 2048;
        
        int MR = 6;
        int NR = 16;

        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int jc = 0; jc < N; jc += NC) {
            for (int pc = 0; pc < K; pc += KC) {
                int j_end = std::min(N, jc + NC);
                int p_end = std::min(K, pc + KC);

                for (int ic = 0; ic < M; ic += MC) {
                    int i_end = std::min(M, ic + MC);

                    // Micro-kernel loops
                    for (int jr = jc; jr < j_end; jr += NR) {
                        for (int ir = ic; ir < i_end; ir += MR) {
                            
                            int m_block = std::min(MR, i_end - ir);
                            int n_block = std::min(NR, j_end - jr);
                            int k_block = p_end - pc;

#if defined(__AVX2__)
                            if (m_block == MR && n_block == NR) {
                                micro_kernel_avx2(
                                    k_block, 
                                    A + ir * lda + pc, lda, 
                                    B + pc * ldb + jr, ldb, 
                                    C + ir * ldc + jr, ldc,
                                    activation);
                            } else 
#endif
                            {
                                micro_kernel_generic(
                                    m_block, n_block, k_block,
                                    A + ir * lda + pc, lda, 
                                    B + pc * ldb + jr, ldb, 
                                    C + ir * ldc + jr, ldc,
                                    activation);
                            }
                        }
                    }
                }
            }
        }
    }

    void CpuEngine::conv2d_fused(
        const float* input, int C_in, int H, int W,
        const float* filter, int C_out, int kH, int kW,
        float* output, int stride, int pad,
        FusedActivation activation)
    {
        // To be implemented using direct convolution (no im2col overhead)
        // or hardware-aware im2col that writes directly to L1 cache
    }

    void CpuEngine::midbits_sgemm(
        int M, int N, int K,
        const uint8_t* A_indices, 
        const float* B,           
        float* C,
        FusedActivation activation)
    {
        // The ultimate CPU MidBits kernel
        // LUT precomputation + matrix multiplication in ONE pass.
    }

    void CpuEngine::qsbits_xnor(
        int M, int K_packed,
        const uint8_t* input, 
        const uint8_t* weights,           
        float* output)
    {
        for (int o = 0; o < M; o++) {
            const uint8_t* w_row = weights + o * K_packed;
            int total_pop = 0;

            int64_t k = 0;
#if defined(__AVX2__) || defined(__AVX512F__)
            for (; k + 7 < K_packed; k += 8) {
                uint64_t xnor = ~(*(const uint64_t*)(input + k) ^ *(const uint64_t*)(w_row + k));
#if defined(_MSC_VER)
                total_pop += (int)__popcnt64(xnor);
#else
                total_pop += __builtin_popcountll(xnor);
#endif
            }
#endif
            for (; k < K_packed; k++) {
                uint8_t x = ~(input[k] ^ w_row[k]);
#if defined(_MSC_VER)
                total_pop += (int)__popcnt16(x); // cast
#else
                total_pop += __builtin_popcount(x);
#endif
            }
            output[o] = (float)(2 * total_pop - K_packed * 8);
        }
    }

    void CpuEngine::qsbits_xnor_scaled(
        int M, int K_packed,
        const uint8_t* input,
        const uint8_t* weights,
        const float* scales,
        float input_scale,
        int group_size,
        float* output)
    {
        const int g_packed = group_size / 8;
        const int n_groups = K_packed / g_packed;

        for (int o = 0; o < M; o++) {
            const uint8_t* w_row = weights + o * K_packed;
            const float* s_row = scales + o * n_groups;
            float acc = 0.0f;

            for (int g = 0; g < n_groups; g++) {
                const int base = g * g_packed;
                int group_pop = 0;

                int64_t k = 0;
#if defined(__AVX2__) || defined(__AVX512F__)
                for (; k + 7 < g_packed; k += 8) {
                    uint64_t xnor = ~(*(const uint64_t*)(input + base + k) ^
                                      *(const uint64_t*)(w_row + base + k));
#if defined(_MSC_VER)
                    group_pop += (int)__popcnt64(xnor);
#else
                    group_pop += __builtin_popcountll(xnor);
#endif
                }
#endif
                for (; k < g_packed; k++) {
                    uint8_t x = ~(input[base + k] ^ w_row[base + k]);
#if defined(_MSC_VER)
                    group_pop += (int)__popcnt16(x);
#else
                    group_pop += __builtin_popcount(x);
#endif
                }

                float val = (float)(2 * group_pop - group_size);
                acc += val * s_row[g];
            }

            output[o] = acc * input_scale;
        }
    }

    void CpuEngine::flux_sgemm(int Batch, int M, int K_bytes, const uint8_t* flux_rows, const uint8_t* query_batch, float* output_fp32) {
        size_t K_uint64 = K_bytes / 8; 
        
        #pragma omp parallel for schedule(dynamic)
        for (int b = 0; b < Batch; ++b) {
            const uint8_t* Q_b = query_batch + (b * K_bytes);
            float* out_b = output_fp32 + (b * M);
            const uint64_t* Q64 = reinterpret_cast<const uint64_t*>(Q_b);
            
            for (int i = 0; i < M; ++i) {
                const uint8_t* F_row = flux_rows + (i * K_bytes);
                const uint64_t* F64 = reinterpret_cast<const uint64_t*>(F_row);
                
                int total_voltage = 0;
                size_t k = 0;

#if defined(__AVX2__) || defined(__AVX512F__)
                for (; k + 3 < K_uint64; k += 4) {
                    __m256i q_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Q64 + k));
                    __m256i f_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(F64 + k));
                    __m256i collision = _mm256_and_si256(q_vec, f_vec);
                    
                    uint64_t cols[4];
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(cols), collision);
                    
#if defined(_MSC_VER)
                    total_voltage += (int)__popcnt64(cols[0]) + (int)__popcnt64(cols[1]) + 
                                     (int)__popcnt64(cols[2]) + (int)__popcnt64(cols[3]);
#else
                    total_voltage += __builtin_popcountll(cols[0]) + __builtin_popcountll(cols[1]) +
                                     __builtin_popcountll(cols[2]) + __builtin_popcountll(cols[3]);
#endif
                }
#endif
                for (; k < K_uint64; ++k) {
                    uint64_t collision = Q64[k] & F64[k];
#if defined(_MSC_VER)
                    total_voltage += (int)__popcnt64(collision);
#else
                    total_voltage += __builtin_popcountll(collision);
#endif
                }
                
                for (size_t rb = k * 8; rb < (size_t)K_bytes; ++rb) {
                    uint8_t collision = Q_b[rb] & F_row[rb];
#if defined(_MSC_VER)
                    total_voltage += (int)__popcnt16(collision);
#else
                    total_voltage += __builtin_popcount(collision);
#endif
                }

                out_b[i] = static_cast<float>(total_voltage);
            }
        }
    }

Engine* create_best_hlas_engine() {
    // In the future, this will return OpenCL/CUDA engine if requested
    return new CpuEngine();
}

} // namespace hlas
} // namespace mecan
