// MecanTensor HLAS: Universal Hybrid Router
// Dynamically profiles the hardware and distributes workload:
// - Small layers/batches -> CPU (low latency)
// - Large layers/batches -> GPU (high throughput)
// - NPUs -> offload specific convolutions

#include "hlas.h"
#include <iostream>

namespace mecan {
namespace hlas {

HybridEngine::HybridEngine() {
    cpu_ = new CpuEngine();
    gpu_ = new GpuEngine();
    npu_ = new NpuEngine();
    calibrate();
}

HybridEngine::~HybridEngine() {
    delete cpu_;
    delete gpu_;
    delete npu_;
}

void HybridEngine::calibrate() {
    // Hardware info should be fetched via the C-API by Python, not printed to stdout.
}

DeviceInfo HybridEngine::get_device_info() const {
    return {"Universal Hybrid Router", BackendType::HYBRID, 0, 0};
}

void HybridEngine::sgemm(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc, FusedActivation act) {
    // ROUTING LOGIC:
    // If matrix is huge, route to GPU. Otherwise CPU.
    if (M > 1024 && N > 1024) {
        gpu_->sgemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, act);
    } else {
        cpu_->sgemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, act);
    }
}

void HybridEngine::conv2d_fused(const float* input, int C_in, int H, int W, const float* filter, int C_out, int kH, int kW, float* output, int stride, int pad, FusedActivation act) {
    if (H > 128 && W > 128) {
        gpu_->conv2d_fused(input, C_in, H, W, filter, C_out, kH, kW, output, stride, pad, act);
    } else {
        cpu_->conv2d_fused(input, C_in, H, W, filter, C_out, kH, kW, output, stride, pad, act);
    }
}

void HybridEngine::midbits_sgemm(int M, int N, int K, const uint8_t* A_indices, const float* B, float* C, FusedActivation act) {
    if (M > 2048) {
        gpu_->midbits_sgemm(M, N, K, A_indices, B, C, act);
    } else {
        cpu_->midbits_sgemm(M, N, K, A_indices, B, C, act);
    }
}

void HybridEngine::qsbits_xnor(int M, int K_packed, const uint8_t* input, const uint8_t* weights, float* output) {
    cpu_->qsbits_xnor(M, K_packed, input, weights, output);
}

void HybridEngine::qsbits_xnor_scaled(int M, int K_packed, const uint8_t* input, const uint8_t* weights, const float* scales, float input_scale, int group_size, float* output) {
    cpu_->qsbits_xnor_scaled(M, K_packed, input, weights, scales, input_scale, group_size, output);
}

void HybridEngine::flux_sgemm(int Batch, int M, int K_bytes, const uint8_t* flux_rows, const uint8_t* query_batch, float* output_fp32) {
    if (Batch > 16 || M > 4096) {
        gpu_->flux_sgemm(Batch, M, K_bytes, flux_rows, query_batch, output_fp32);
    } else {
        cpu_->flux_sgemm(Batch, M, K_bytes, flux_rows, query_batch, output_fp32);
    }
}

// Global Factory Method
Engine* create_hybrid_engine() {
    return new HybridEngine();
}

namespace {
    Engine* global_engine = nullptr;
    std::once_flag engine_init_flag;
}

// Global Singleton Accessor
Engine* get_engine() {
    std::call_once(engine_init_flag, []() {
        global_engine = create_hybrid_engine();
    });
    return global_engine;
}

} // namespace hlas
} // namespace mecan
