// MecanTensor: FluxCompiler Implementation
#include "mecan/ops/fluxbits.h"
#include <cmath>
#include <stdexcept>
#include <random>

namespace mecan {
namespace fluxbits {

    // Simple fast hash to map (j, hash_idx) to a specific bit in the global Flux Space.
    // We use a combination of FNV-1a and bitwise mixing.
    inline size_t flux_hash(size_t j, int hash_idx, size_t max_bits) {
        uint64_t hash = 14695981039346656037ULL;
        hash ^= (uint64_t)j;
        hash *= 1099511628211ULL;
        hash ^= (uint64_t)hash_idx;
        hash *= 1099511628211ULL;
        
        // Final avalanche mix
        hash ^= hash >> 33;
        hash *= 0xff51afd7ed558ccdULL;
        hash ^= hash >> 33;
        hash *= 0xc4ceb9fe1a85ec53ULL;
        hash ^= hash >> 33;

        return static_cast<size_t>(hash % max_bits);
    }

    std::vector<uint8_t> FluxCompiler::compile_flux_rows(
        const float* dense_weights, 
        size_t M, size_t N,
        int d_hashes) 
    {
        // Each output neuron 'i' in M looks at N inputs.
        // We compress the N inputs into K bits for that specific neuron.
        // For 0.45 bits/param, K = 0.45 * N.
        size_t K_bits = static_cast<size_t>(0.45 * N);
        
        // Ensure K is a multiple of 8 for byte alignment
        if (K_bits % 8 != 0) {
            K_bits += (8 - (K_bits % 8));
        }
        size_t K_bytes = K_bits / 8;

        std::vector<uint8_t> flux_matrix(M * K_bytes, 0);

        for (size_t i = 0; i < M; ++i) {
            uint8_t* row_ptr = flux_matrix.data() + (i * K_bytes);

            for (size_t j = 0; j < N; ++j) {
                float weight = dense_weights[i * N + j];
                
                // Absolute threshold for existence of a synapse.
                // In a trained TSSR model, weights > threshold are critical paths.
                if (std::abs(weight) > 0.05f) { // Threshold can be parameterized
                    // Map this connection into the neuron's specific FluxRow using d_hashes
                    for (int h = 0; h < d_hashes; ++h) {
                        size_t bit_idx = flux_hash(j, h, K_bits);
                        size_t byte_idx = bit_idx / 8;
                        size_t bit_offset = bit_idx % 8;
                        
                        row_ptr[byte_idx] |= (1 << bit_offset);
                    }
                }
            }
        }

        return flux_matrix;
    }

} // namespace fluxbits
} // namespace mecan
