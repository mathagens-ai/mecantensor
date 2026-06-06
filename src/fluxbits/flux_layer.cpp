// MecanTensor: FluxLayer v5 — Bounded Loss + 99% Parameter Intelligence
//   1. Loss ∈ [0, 5.0] — HARD BOUNDED. 5.0 = no knowledge, 0.0 = perfect
//   2. Loss < 0.1111 = overfitting warning
//   3. Target convergence zone: 1.0 - 2.0 (good learning)
//   4. Parameter utilization: 98-99% minimum
//   5. Every parameter learns with maximum intelligence per bit

#include "mecan/ops/fluxbits.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <omp.h>
#include <iostream>
#include <cstring>
#include <numeric>

#if defined(_MSC_VER)
#include <intrin.h>
#include <nmmintrin.h>
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif

namespace mecan {
namespace fluxbits {

    static inline size_t flux_hash(size_t j, int hash_idx, size_t max_bits) {
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

    // Affine Calibrator v5 — Proper output range matching

    void AffineCalibrator::init(size_t M, size_t K_bits, float bloom_density) {
        gamma.resize(M);
        beta.resize(M, 0.0f);

        // mu_expected: the ACTUAL mean popcount we'll see with this bloom density
        // For ~10% input density and given bloom_density:
        //   expected = K_bits * P(Q_bit=1) * P(F_bit=1)
        // P(Q_bit=1) depends on how many input neurons hash to each bit
        // Empirically: with d=2 hashes, ~10% input, K_bits bits: mu ≈ K * density * input_density
        mu_expected = static_cast<float>(K_bits) * bloom_density * 0.1f;

        // CRITICAL FIX: gamma must map popcount variance to TARGET variance
        // Target values are in [-2, 2] range with stddev ≈ 1.4
        // Popcount variance ≈ K_bits * p * (1-p)
        float p = bloom_density * 0.1f;
        float popcount_stddev = std::sqrt(static_cast<float>(K_bits) * p * (1.0f - p));
        if (popcount_stddev < 0.01f) popcount_stddev = 1.0f;

        // Map popcount stddev → target stddev (≈1.0 for normalized output)
        float target_stddev = 1.0f;
        float init_gamma = target_stddev / popcount_stddev;

        for (size_t i = 0; i < M; ++i) {
            gamma[i] = init_gamma;
        }
    }

    void AffineCalibrator::apply(float* output, size_t Batch, size_t M) const {
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < (int)Batch; ++b) {
            float* out_b = output + b * M;
            for (size_t i = 0; i < M; ++i) {
                out_b[i] = gamma[i] * (out_b[i] - mu_expected) + beta[i];
            }
        }
    }

    // FluxLayer v5

    FluxLayer::FluxLayer(size_t M, size_t N, int d_hashes, float threshold)
        : M_(M), N_(N), d_hashes_(d_hashes), threshold_(threshold)
    {
        K_bits_ = static_cast<size_t>(0.45 * N);
        if (K_bits_ % 8 != 0) K_bits_ += (8 - (K_bits_ % 8));
        K_bytes_ = K_bits_ / 8;

        // Initialize weights with He-like distribution
        latent_weights_.resize(M * N);
        std::mt19937 gen(42);
        float stddev = std::sqrt(2.0f / static_cast<float>(N));
        std::normal_distribution<float> dist(0.0f, stddev);
        for (size_t i = 0; i < M * N; ++i) {
            latent_weights_[i] = dist(gen);
        }

        latent_grad_.resize(M * N, 0.0f);
        lgc_momentum_.resize(M * N, 0.0f);
        lgc_curvature_.resize(M * N, 1.0f);
        vitality_.resize(M * N, 1.0f);

        // Pre-compute hash map
        hash_map_.resize(M * N * d_hashes);
        #pragma omp parallel for schedule(static)
        for (int64_t idx = 0; idx < (int64_t)(M * N); ++idx) {
            size_t j = idx % N;
            for (int h = 0; h < d_hashes; ++h) {
                hash_map_[idx * d_hashes + h] = flux_hash(j, h, K_bits_);
            }
        }

        size_t active_synapses = 0;
        for (size_t i = 0; i < M * N; ++i) {
            if (std::abs(latent_weights_[i]) > threshold_) active_synapses++;
        }
        float bloom_density = 1.0f - std::pow(1.0f - 1.0f / static_cast<float>(K_bits_),
                                                static_cast<float>(active_synapses * d_hashes_) / static_cast<float>(M));
        calibrator_.init(M, K_bits_, bloom_density);
        rehash();

        float util = (float)active_synapses / (float)(M * N) * 100.0f;
        std::cout << "[FluxLayer] M=" << M << " N=" << N
                  << " K=" << K_bits_ << " Flux=" << (flux_rows_.size() / 1024.0 / 1024.0) << "MB"
                  << " Util=" << util << "%\n";
    }

    void FluxLayer::rehash() {
        flux_rows_ = FluxCompiler::compile_flux_rows(
            latent_weights_.data(), M_, N_, d_hashes_
        );
    }

    void FluxLayer::forward(int Batch, const uint8_t* input_packed, float* output) {
        size_t K_uint64 = K_bytes_ / 8;

        #pragma omp parallel for schedule(dynamic)
        for (int b = 0; b < Batch; ++b) {
            const uint8_t* Q_b = input_packed + (b * K_bytes_);
            float* out_b = output + (b * M_);
            const uint64_t* Q64 = reinterpret_cast<const uint64_t*>(Q_b);

            for (size_t i = 0; i < M_; ++i) {
                const uint8_t* F_row = flux_rows_.data() + (i * K_bytes_);
                const uint64_t* F64 = reinterpret_cast<const uint64_t*>(F_row);

                int total_voltage = 0;
                size_t k = 0;
                for (; k < K_uint64; ++k) {
                    uint64_t collision = Q64[k] & F64[k];
#if defined(_MSC_VER)
                    total_voltage += (int)__popcnt64(collision);
#else
                    total_voltage += __builtin_popcountll(collision);
#endif
                }
                for (size_t rb = k * 8; rb < K_bytes_; ++rb) {
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
        calibrator_.apply(output, Batch, M_);
    }

    // Backward v5 — clean STE, no artificial injection

    void FluxLayer::backward(int Batch, const float* grad_output, const uint8_t* input_packed) {
        std::vector<float> neuron_grad(M_, 0.0f);
        for (size_t i = 0; i < M_; ++i) {
            float sum = 0.0f;
            for (int b = 0; b < Batch; ++b) sum += grad_output[b * M_ + i];
            neuron_grad[i] = sum * calibrator_.gamma[i];
        }

        std::vector<float> input_freq(N_, 0.0f);
        #pragma omp parallel
        {
            std::vector<float> lf(N_, 0.0f);
            #pragma omp for schedule(static)
            for (int b = 0; b < Batch; ++b) {
                const uint8_t* Q_b = input_packed + (b * K_bytes_);
                for (size_t j = 0; j < N_; ++j) {
                    bool active = true;
                    for (int h = 0; h < d_hashes_; ++h) {
                        size_t bi = hash_map_[j * d_hashes_ + h];
                        if (bi / 8 < K_bytes_ && !(Q_b[bi / 8] & (1 << (bi % 8)))) { active = false; break; }
                    }
                    if (active) lf[j] += 1.0f;
                }
            }
            #pragma omp critical
            { for (size_t j = 0; j < N_; ++j) input_freq[j] += lf[j]; }
        }
        float binv = 1.0f / static_cast<float>(Batch);
        for (size_t j = 0; j < N_; ++j) input_freq[j] *= binv;

        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)M_; ++i) {
            float g_i = neuron_grad[i];
            if (std::abs(g_i) < 1e-15f) continue;
            const uint8_t* F_row = flux_rows_.data() + (i * K_bytes_);
            float* gr = latent_grad_.data() + (i * N_);
            float* vr = vitality_.data() + (i * N_);

            for (size_t j = 0; j < N_; ++j) {
                if (input_freq[j] > 0.0f) {
                    gr[j] += g_i * input_freq[j];
                    vr[j] = std::min(vr[j] + 0.05f, 2.0f);
                } else {
                    vr[j] *= 0.998f;
                }
            }
        }

        // Beta gradient
        for (size_t i = 0; i < M_; ++i) {
            float db = 0.0f;
            for (int b = 0; b < Batch; ++b) db += grad_output[b * M_ + i];
            calibrator_.beta[i] -= 0.001f * db / static_cast<float>(Batch);
        }
    }

    void FluxLayer::lgc_step(float lr, float momentum_decay) {
        float gnorm = 0.0f;
        #pragma omp parallel for reduction(+:gnorm)
        for (int64_t i = 0; i < (int64_t)(M_ * N_); ++i) gnorm += latent_grad_[i] * latent_grad_[i];
        gnorm = std::sqrt(gnorm / (float)(M_ * N_));
        float clip = (gnorm > 1.0f) ? (1.0f / gnorm) : 1.0f;

        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)(M_ * N_); ++i) {
            float g = latent_grad_[i] * clip;
            lgc_momentum_[i] = lgc_momentum_[i] * momentum_decay + g;
            lgc_curvature_[i] = lgc_curvature_[i] * momentum_decay + g * g;
            float safe_curv = std::sqrt(lgc_curvature_[i]) + 1e-8f;
            float delta = lgc_momentum_[i] / safe_curv;
            latent_weights_[i] *= (1.0f - lr * 1e-4f);
            latent_weights_[i] -= lr * delta;
        }
        rehash();
        // mu_expected is NOT updated here anymore — it's set by calibrate_output
        // and stays stable during training. The beta parameter handles output drift.
    }

    void FluxLayer::zero_grad() {
        std::fill(latent_grad_.begin(), latent_grad_.end(), 0.0f);
    }

    // Empirical Output Calibration
    // Runs a raw forward pass (without calibrator) to measure the ACTUAL mean
    // and stddev of popcount outputs, then sets mu_expected and gamma so that
    // the calibrated output is zero-centered with unit variance.
    // THIS MUST BE CALLED ONCE BEFORE TRAINING with a representative batch.
    void FluxLayer::calibrate_output(int Batch, const uint8_t* input_packed) {
        size_t K_uint64 = K_bytes_ / 8;
        
        // Run raw popcount forward (no calibrator)
        std::vector<float> raw_output(Batch * M_, 0.0f);
        
        for (int b = 0; b < Batch; ++b) {
            const uint8_t* Q_b = input_packed + (b * K_bytes_);
            const uint64_t* Q64 = reinterpret_cast<const uint64_t*>(Q_b);

            for (size_t i = 0; i < M_; ++i) {
                const uint8_t* F_row = flux_rows_.data() + (i * K_bytes_);
                const uint64_t* F64 = reinterpret_cast<const uint64_t*>(F_row);

                int voltage = 0;
                size_t k = 0;
                for (; k < K_uint64; ++k) {
                    uint64_t collision = Q64[k] & F64[k];
#if defined(_MSC_VER)
                    voltage += (int)__popcnt64(collision);
#else
                    voltage += __builtin_popcountll(collision);
#endif
                }
                for (size_t rb = k * 8; rb < K_bytes_; ++rb) {
                    uint8_t collision = Q_b[rb] & F_row[rb];
#if defined(_MSC_VER)
                    voltage += (int)__popcnt16(collision);
#else
                    voltage += __builtin_popcount(collision);
#endif
                }
                raw_output[b * M_ + i] = static_cast<float>(voltage);
            }
        }

        // Compute per-neuron mean and overall statistics
        float global_mean = 0.0f;
        float global_var = 0.0f;
        size_t total = Batch * M_;

        for (size_t idx = 0; idx < total; ++idx) {
            global_mean += raw_output[idx];
        }
        global_mean /= static_cast<float>(total);

        for (size_t idx = 0; idx < total; ++idx) {
            float d = raw_output[idx] - global_mean;
            global_var += d * d;
        }
        global_var /= static_cast<float>(total);
        float global_stddev = std::sqrt(global_var);
        if (global_stddev < 0.01f) global_stddev = 1.0f;

        // Set calibrator: output = gamma * (raw - mu) + beta
        // We want output mean ≈ 0, stddev ≈ 1
        calibrator_.mu_expected = global_mean;
        float target_stddev = 1.0f;
        float gamma_val = target_stddev / global_stddev;
        
        for (size_t i = 0; i < M_; ++i) {
            calibrator_.gamma[i] = gamma_val;
            calibrator_.beta[i] = 0.0f;
        }

        std::cout << "[CALIBRATE] Raw popcount: mean=" << global_mean 
                  << " stddev=" << global_stddev 
                  << " → gamma=" << gamma_val << "\n";
    }

    std::vector<Tensor> FluxBackward::apply(const std::vector<Tensor>& grad_outputs) {
        layer_->backward(batch_, grad_outputs[0].data_ptr<float>(), saved_input_.data());
        return {};
    }

} // namespace fluxbits
} // namespace mecan

