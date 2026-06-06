// MecanTensor HLAS (Hardware Linear Algebra Subroutines)
// Universal Core Header (Full Library)

#ifndef MECAN_HLAS_H
#define MECAN_HLAS_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

namespace mecan {
namespace hlas {

// ─── HARDWARE ARCHITECTURE DETECTION ───────────────────────────────────────
enum class CpuArch { GENERIC, X86_AVX2, X86_AVX512, ARM_NEON, ARM_SVE };
enum class BackendType { CPU, GPU_OPENCL, GPU_CUDA, NPU_DIRECTML, HYBRID };

struct CacheTopology {
    size_t l1_data_size;    
    size_t l2_size;         
    size_t l3_size;         
    size_t cacheline_size;  
};

struct DeviceInfo {
    std::string name;
    BackendType type;
    size_t max_vram;
    int compute_units;
};

enum class FusedActivation { NONE, RELU, SILU, GELU };

// ─── UNIVERSAL INTERFACE ───────────────────────────────────────────────────
class Engine {
public:
    virtual ~Engine() = default;

    virtual void calibrate() = 0;
    virtual DeviceInfo get_device_info() const = 0;

    virtual void sgemm(
        int M, int N, int K,
        float alpha, const float* A, int lda,
        const float* B, int ldb,
        float beta, float* C, int ldc,
        FusedActivation activation = FusedActivation::NONE
    ) = 0;

    virtual void conv2d_fused(
        const float* input, int C_in, int H, int W,
        const float* filter, int C_out, int kH, int kW,
        float* output, int stride, int pad,
        FusedActivation activation = FusedActivation::NONE
    ) = 0;

    virtual void midbits_sgemm(
        int M, int N, int K,
        const uint8_t* A_indices, 
        const float* B,           
        float* C,
        FusedActivation activation = FusedActivation::NONE
    ) = 0;
    
    virtual void qsbits_xnor(
        int M, int K_packed,
        const uint8_t* input, 
        const uint8_t* weights,           
        float* output
    ) = 0;

    virtual void qsbits_xnor_scaled(
        int M, int K_packed,
        const uint8_t* input, 
        const uint8_t* weights,
        const float* scales,
        float input_scale,
        int group_size,
        float* output
    ) = 0;

    virtual void flux_sgemm(
        int Batch, int M, int K_bytes,
        const uint8_t* flux_rows,
        const uint8_t* query_batch,
        float* output_fp32
    ) = 0;
};

// ─── COMPONENT ENGINES ─────────────────────────────────────────────────────

class CpuEngine : public Engine {
public:
    CpuEngine();
    void calibrate() override;
    DeviceInfo get_device_info() const override;
    void sgemm(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc, FusedActivation act) override;
    void conv2d_fused(const float* input, int C_in, int H, int W, const float* filter, int C_out, int kH, int kW, float* output, int stride, int pad, FusedActivation act) override;
    void midbits_sgemm(int M, int N, int K, const uint8_t* A_indices, const float* B, float* C, FusedActivation act) override;
    void qsbits_xnor(int M, int K_packed, const uint8_t* input, const uint8_t* weights, float* output) override;
    void qsbits_xnor_scaled(int M, int K_packed, const uint8_t* input, const uint8_t* weights, const float* scales, float input_scale, int group_size, float* output) override;
    void flux_sgemm(int Batch, int M, int K_bytes, const uint8_t* flux_rows, const uint8_t* query_batch, float* output_fp32) override;
private:
    CpuArch arch_;
    CacheTopology cache_;
};

class GpuEngine : public Engine {
public:
    GpuEngine();
    ~GpuEngine() override;
    void calibrate() override;
    DeviceInfo get_device_info() const override;
    void sgemm(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc, FusedActivation act) override;
    void conv2d_fused(const float* input, int C_in, int H, int W, const float* filter, int C_out, int kH, int kW, float* output, int stride, int pad, FusedActivation act) override;
    void midbits_sgemm(int M, int N, int K, const uint8_t* A_indices, const float* B, float* C, FusedActivation act) override;
    void qsbits_xnor(int M, int K_packed, const uint8_t* input, const uint8_t* weights, float* output) override;
    void qsbits_xnor_scaled(int M, int K_packed, const uint8_t* input, const uint8_t* weights, const float* scales, float input_scale, int group_size, float* output) override;
    void flux_sgemm(int Batch, int M, int K_bytes, const uint8_t* flux_rows, const uint8_t* query_batch, float* output_fp32) override;
private:
    bool initialized_;
    DeviceInfo info_;
    // OpenCL Handles (stored as void* to avoid heavy CL headers in interface)
    void* context_;
    void* queue_;
    void* program_;
};

class NpuEngine : public Engine {
public:
    NpuEngine();
    void calibrate() override;
    DeviceInfo get_device_info() const override;
    void sgemm(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc, FusedActivation act) override;
    void conv2d_fused(const float* input, int C_in, int H, int W, const float* filter, int C_out, int kH, int kW, float* output, int stride, int pad, FusedActivation act) override;
    void midbits_sgemm(int M, int N, int K, const uint8_t* A_indices, const float* B, float* C, FusedActivation act) override;
    void qsbits_xnor(int M, int K_packed, const uint8_t* input, const uint8_t* weights, float* output) override;
    void qsbits_xnor_scaled(int M, int K_packed, const uint8_t* input, const uint8_t* weights, const float* scales, float input_scale, int group_size, float* output) override;
    void flux_sgemm(int Batch, int M, int K_bytes, const uint8_t* flux_rows, const uint8_t* query_batch, float* output_fp32) override;
private:
    DeviceInfo info_;
};

class HybridEngine : public Engine {
public:
    HybridEngine();
    ~HybridEngine() override;
    void calibrate() override;
    DeviceInfo get_device_info() const override;
    void sgemm(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc, FusedActivation act) override;
    void conv2d_fused(const float* input, int C_in, int H, int W, const float* filter, int C_out, int kH, int kW, float* output, int stride, int pad, FusedActivation act) override;
    void midbits_sgemm(int M, int N, int K, const uint8_t* A_indices, const float* B, float* C, FusedActivation act) override;
    void qsbits_xnor(int M, int K_packed, const uint8_t* input, const uint8_t* weights, float* output) override;
    void qsbits_xnor_scaled(int M, int K_packed, const uint8_t* input, const uint8_t* weights, const float* scales, float input_scale, int group_size, float* output) override;
    void flux_sgemm(int Batch, int M, int K_bytes, const uint8_t* flux_rows, const uint8_t* query_batch, float* output_fp32) override;
private:
    CpuEngine* cpu_;
    GpuEngine* gpu_;
    NpuEngine* npu_;
};

// Factory to get the universal hybrid engine
Engine* create_hybrid_engine();

// Global singleton accessor
Engine* get_engine();

} // namespace hlas
} // namespace mecan

#endif // MECAN_HLAS_H
