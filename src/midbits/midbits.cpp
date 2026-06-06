// MidBits: The 0.75-Bit Matrix Engine
// Implements the Multi-Path Block-Palette system.
//
// 1. Path A (Pre-Compute): Generates the 256-element LUT for a chunk of 16 inputs.
// 2. Path B (Branchless Unpack): Looks up the precomputed answers using the 8-bit weights.
// 3. Path C (Accumulation): Sums the results.

#include <immintrin.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include "midbits_codebook.h"
#include "mecan/midbits/midbits.h"
#include "../hlas/hlas.h"

namespace mecan {
namespace midbits {

    // ─── Path A: Activation Pre-Compute (The LUT Generator) ───────────────
    // Takes 16 float32 activations and computes their dot-product against
    // all 256 sparse patterns in the Codebook.
    // Because each pattern has exactly 2 non-zero elements (+1 or -1),
    // this requires ZERO multiplications, only additions/subtractions.
    void precompute_lut(const float* x_chunk_16, float* lut_256) {
        // In a fully optimized version, we wouldn't loop through the raw CODEBOOK array.
        // We would use an AVX2 vectorized gather/add. For this prototype, we show the math.
        for (int i = 0; i < 256; i++) {
            float sum = 0.0f;
            const int8_t* pattern = &CODEBOOK[i * 16];
            
            // Unrolled dot product for exactly 16 elements
            sum += x_chunk_16[0]  * pattern[0];
            sum += x_chunk_16[1]  * pattern[1];
            sum += x_chunk_16[2]  * pattern[2];
            sum += x_chunk_16[3]  * pattern[3];
            sum += x_chunk_16[4]  * pattern[4];
            sum += x_chunk_16[5]  * pattern[5];
            sum += x_chunk_16[6]  * pattern[6];
            sum += x_chunk_16[7]  * pattern[7];
            sum += x_chunk_16[8]  * pattern[8];
            sum += x_chunk_16[9]  * pattern[9];
            sum += x_chunk_16[10] * pattern[10];
            sum += x_chunk_16[11] * pattern[11];
            sum += x_chunk_16[12] * pattern[12];
            sum += x_chunk_16[13] * pattern[13];
            sum += x_chunk_16[14] * pattern[14];
            sum += x_chunk_16[15] * pattern[15];
            
            lut_256[i] = sum;
        }
    }

    // ─── Path B & C: LUT-Based Matrix Multiplication ──────────────────────
    // M = Output dimensions (Neurons)
    // K = Input dimensions
    // W_midbits = Matrix of size (M, K/16), stored as uint8_t!
    // X = Input vector of size (K)
    // Y = Output vector of size (M)
    void matvec_0_75b(const uint8_t* W_midbits, const float* X, float* Y, int M, int K) {
        // Ensure K is a multiple of 16
        if (K % 16 != 0) return;
        
        // Pass directly to the Universal HLAS engine with N=1 (vector mode)
        // MidBits matrix dimensions are transposed relative to dense standard
        mecan::hlas::get_engine()->midbits_sgemm(M, 1, K, W_midbits, X, Y, mecan::hlas::FusedActivation::NONE);
    }

} // namespace midbits
} // namespace mecan
