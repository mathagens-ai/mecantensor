// This module breaks the Shannon limit by converting linear matrix weights
// into a Probabilistic Hyper-Dimensional Bloom Tensor, executed exclusively
// via SIMD Bitwise operations (AND + Popcount).
//
// Training Support:
//   - Affine Calibrator: shifts popcount integers into zero-centered FP32 space
//   - STE Backward:      routes gradients to latent FP32 weights via LGC
//   - Re-Hash on step:   FluxCompiler rebuilds Bloom Tensor after each update

#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include "../tensor.h"
#include "../autograd/node.h"

namespace mecan {
namespace fluxbits {

    // ─── Flux Compiler ────────────────────────────────────────────────────────
    class FluxCompiler {
    public:
        // Compresses a standard Dense Weight Matrix (FP32/FP16) into a Flux Bloom Tensor.
        // M: Output features
        // N: Input features
        // target_compression_ratio: e.g., 0.45 (bits per parameter)
        // d_hashes: Number of hash collisions per synapse
        static std::vector<uint8_t> compile_weights(
            const float* dense_weights, 
            size_t M, size_t N, 
            float target_compression_ratio = 0.45f,
            int d_hashes = 2
        );

        // Advanced compilation: Instead of a flat 1D Bloom filter, we compile
        // specific "Flux Rows" per neuron to make execution cache-friendly.
        // Returns the packed bit arrays for each output neuron.
        static std::vector<uint8_t> compile_flux_rows(
            const float* dense_weights, 
            size_t M, size_t N,
            int d_hashes = 2
        );
    };

    // ─── Flux Execution Kernel ────────────────────────────────────────────────
    class FluxKernel {
    public:
        // Executes a forward pass using the Probabilistic Flux Rows.
        // M: Output features
        // N: Input features (must be quantized to binary 0/1 bits packed into uint8_t)
        // d_hashes: Number of hash collisions per synapse
        // flux_rows: The precompiled Bloom Tensor matrix
        // input_packed: The activation input packed into bits
        // output_fp32: The resulting voltage sums per neuron
        static void execute_forward_avx2(
            size_t M, size_t N, 
            int d_hashes,
            const uint8_t* flux_rows,
            const uint8_t* input_packed,
            float* output_fp32
        );
    };

    // ─── Affine Calibrator ────────────────────────────────────────────────────
    // Projection 1: Maps raw popcount integers back to zero-centered FP32 space
    //   output_fp32[i] = gamma[i] * (raw_popcount[i] - mu_expected) + beta[i]
    //
    // gamma, beta are learnable parameters (updated by LGC during backward)
    // mu_expected = expected collision count based on Bloom density
    struct AffineCalibrator {
        std::vector<float> gamma;   // per-neuron scale  (M elements)
        std::vector<float> beta;    // per-neuron shift   (M elements)
        float mu_expected;          // expected collision mean

        AffineCalibrator() : mu_expected(0.0f) {}

        void init(size_t M, size_t K_bits, float bloom_density);

        // In-place: transform raw popcount output into calibrated FP32
        void apply(float* output, size_t Batch, size_t M) const;
    };

    // ─── FluxLayer ────────────────────────────────────────────────────────────
    // A complete trainable layer that:
    //   Forward:  hash latent weights -> flux_rows -> POPCNT -> Affine Calibrate
    //   Backward: STE gradient -> route to latent FP32 weights -> LGC step
    //   Re-Hash:  after LGC updates latent weights, rebuild flux_rows
    class FluxLayer {
    public:
        FluxLayer(size_t M, size_t N, int d_hashes = 2, float threshold = 0.05f);

        // Forward pass: takes packed binary input, returns calibrated FP32 output
        // input_packed: [Batch, K_bytes] packed bits
        // output:       [Batch, M] calibrated FP32
        void forward(int Batch, const uint8_t* input_packed, float* output);

        // Backward pass: STE gradient routing
        // grad_output:  [Batch, M] incoming gradients from loss
        // input_packed: [Batch, K_bytes] saved from forward
        // Writes gradients into latent_grad_ for LGC to consume
        void backward(int Batch, const float* grad_output, const uint8_t* input_packed);

        // LGC Optimizer step: updates latent_weights_ using accumulated gradients
        // then re-hashes to rebuild flux_rows_
        void lgc_step(float lr, float momentum_decay);

        // Zero accumulated gradients
        void zero_grad();

        // Calibrate the Affine Calibrator using REAL data
        // Runs a forward pass to measure actual popcount mean/stddev,
        // then sets gamma and mu_expected so output is zero-centered with unit variance
        void calibrate_output(int Batch, const uint8_t* input_packed);

        // Accessors
        const std::vector<uint8_t>& flux_rows() const { return flux_rows_; }
        const std::vector<float>& latent_weights() const { return latent_weights_; }
        const std::vector<float>& vitality() const { return vitality_; }
        size_t M() const { return M_; }
        size_t N() const { return N_; }
        size_t K_bits() const { return K_bits_; }
        size_t K_bytes() const { return K_bytes_; }

    private:
        size_t M_;               // output features
        size_t N_;               // input features
        size_t K_bits_;          // bits per FluxRow (0.45 * N, aligned)
        size_t K_bytes_;         // K_bits / 8
        int d_hashes_;           // hash functions per synapse
        float threshold_;        // weight magnitude threshold for synapse existence

        // Latent Space (continuous, differentiable — lives in FP32 RAM)
        std::vector<float> latent_weights_;    // [M * N] — the "real" weights
        std::vector<float> latent_grad_;       // [M * N] — accumulated gradients

        // LGC momentum state (per-parameter)
        std::vector<float> lgc_momentum_;      // [M * N]
        std::vector<float> lgc_curvature_;     // [M * N]

        // Ephemeral Execution Proxy (0.45-bit Bloom Tensor — rebuilt each step)
        std::vector<uint8_t> flux_rows_;       // [M * K_bytes]

        // Pre-computed hash map: eliminates per-step hashing in backward
        // hash_map_[idx * d_hashes + h] = bit position for weight idx, hash h
        std::vector<size_t> hash_map_;         // [M * N * d_hashes]

        // Synapse Vitality: tracks parameter contribution over time
        // High vitality (>1.0) = active, contributing parameter
        // Low vitality (<0.3) = dying parameter → gets gradient boost
        std::vector<float> vitality_;          // [M * N]

        // Affine Calibrator (learnable)
        AffineCalibrator calibrator_;

        // Re-hash latent weights into flux_rows_
        void rehash();
    };

    // ─── FluxBackward (Autograd Node) ─────────────────────────────────────────
    // STE: passes gradients through the non-differentiable AND gate
    // by routing them directly to the latent FP32 weight space
    class FluxBackward : public autograd::Node {
    private:
        FluxLayer* layer_;       // non-owning pointer to the layer
        int batch_;
        std::vector<uint8_t> saved_input_;  // input_packed saved from forward
    public:
        FluxBackward(FluxLayer* layer, int batch, const uint8_t* input_packed, size_t input_size)
            : layer_(layer), batch_(batch), saved_input_(input_packed, input_packed + input_size) {}

        std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override;
    };

} // namespace fluxbits
} // namespace mecan

