// MecanTensor: FluxKernel Implementation (AVX2)
#include "mecan/ops/fluxbits.h"
#include <omp.h>
#include <iostream>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace mecan {
namespace fluxbits {

    // Same global hash function required to compile the query vector
    inline size_t flux_hash(size_t j, int hash_idx, size_t max_bits) {
        uint64_t hash = 14695981039346656037ULL;
        hash ^= (uint64_t)j;
        hash *= 1099511628211ULL;
        hash ^= (uint64_t)hash_idx;
        hash *= 1099511628211ULL;
        
        hash ^= hash >> 33;
        hash *= 0xff51afd7ed558ccdULL;
        hash ^= hash >> 33;
        hash *= 0xc4ceb9fe1a85ec53ULL;
        hash ^= hash >> 33;

        return static_cast<size_t>(hash % max_bits);
    }

    void FluxKernel::execute_forward_avx2(
        size_t M, size_t N, 
        int d_hashes,
        const uint8_t* flux_rows,
        const uint8_t* input_packed,
        float* output_fp32)
    {
        // 1. Build the Query Vector Q from the packed input bits
        size_t K_bits = static_cast<size_t>(0.45 * N);
        if (K_bits % 8 != 0) K_bits += (8 - (K_bits % 8));
        size_t K_bytes = K_bits / 8;

        std::vector<uint8_t> query_vector(K_bytes, 0);

        // input_packed is roughly N/8 bytes long.
        // We unpack the bits, and for each active bit (1), we insert into the Query Vector
        size_t N_bytes = (N + 7) / 8;
        for (size_t byte_idx = 0; byte_idx < N_bytes; ++byte_idx) {
            uint8_t val = input_packed[byte_idx];
            if (!val) continue; // Skip zero bytes entirely (Sparsity acceleration!)

            for (int b = 0; b < 8; ++b) {
                if (val & (1 << b)) {
                    size_t j = byte_idx * 8 + b;
                    if (j >= N) break;

                    // Active connection found in input. Hash it into the Query Vector.
                    for (int h = 0; h < d_hashes; ++h) {
                        size_t hash_idx = flux_hash(j, h, K_bits);
                        query_vector[hash_idx / 8] |= (1 << (hash_idx % 8));
                    }
                }
            }
        }

        // 2. AVX2 Popcount Execution (Bitwise AND + Popcount)
        const uint8_t* Q = query_vector.data();
        size_t K_uint64 = K_bytes / 8; // Number of 64-bit blocks to process

        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < (int)M; ++i) {
            const uint8_t* F_row = flux_rows + (i * K_bytes);
            
            int total_voltage = 0;
            const uint64_t* Q64 = reinterpret_cast<const uint64_t*>(Q);
            const uint64_t* F64 = reinterpret_cast<const uint64_t*>(F_row);
            
            size_t k = 0;

#if defined(__AVX2__) || defined(__AVX512F__)
            // Unroll loop for AVX2 speed
            for (; k + 3 < K_uint64; k += 4) {
                // Read 4 chunks of 64-bit (32 bytes = 256 bits total)
                __m256i q_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Q64 + k));
                __m256i f_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(F64 + k));
                
                // Bitwise AND
                __m256i collision = _mm256_and_si256(q_vec, f_vec);
                
                // Extract to 64-bit integers and popcount (since AVX2 lacks native 256-bit popcount)
                uint64_t cols[4];
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(cols), collision);
                
#if defined(_MSC_VER)
                total_voltage += (int)__popcnt64(cols[0]);
                total_voltage += (int)__popcnt64(cols[1]);
                total_voltage += (int)__popcnt64(cols[2]);
                total_voltage += (int)__popcnt64(cols[3]);
#else
                total_voltage += __builtin_popcountll(cols[0]);
                total_voltage += __builtin_popcountll(cols[1]);
                total_voltage += __builtin_popcountll(cols[2]);
                total_voltage += __builtin_popcountll(cols[3]);
#endif
            }
#endif
            // Scalar fallback for remaining elements
            for (; k < K_uint64; ++k) {
                uint64_t collision = Q64[k] & F64[k];
#if defined(_MSC_VER)
                total_voltage += (int)__popcnt64(collision);
#else
                total_voltage += __builtin_popcountll(collision);
#endif
            }
            
            // Any remaining bytes not aligned to 64-bit (usually rare if K_bytes % 8 == 0)
            for (size_t b = k * 8; b < K_bytes; ++b) {
                uint8_t collision = Q[b] & F_row[b];
#if defined(_MSC_VER)
                total_voltage += (int)__popcnt16(collision);
#else
                total_voltage += __builtin_popcount(collision);
#endif
            }

            // Output FP32 Voltage
            output_fp32[i] = static_cast<float>(total_voltage);
        }
    }

} // namespace fluxbits
} // namespace mecan
