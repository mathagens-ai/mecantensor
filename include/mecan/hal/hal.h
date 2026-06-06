#pragma once
// Supports EVERY compute device on the planet:
//   CPU:     x86 (AVX2/AVX-512), ARM (NEON/SVE), RISC-V, MIPS, POWER
//   GPU:     NVIDIA (CUDA), AMD (ROCm/HIP), Intel (oneAPI), Qualcomm (Adreno)
//   NPU:     Google TPU, Intel Movidius, Qualcomm Hexagon, Samsung NPU
//   APU:     AMD APU (shared CPU+GPU memory), Apple Silicon (M1-M4 Unified)
//   FPGA:    Xilinx/AMD, Intel/Altera, Lattice
//   Custom:  Any hardware that exposes OpenCL 1.2 or Vulkan Compute
//
// Design: Runtime dynamic dispatch. No SDK required at compile time.
//   - OpenCL: loaded via LoadLibrary/dlopen at runtime
//   - CUDA:   loaded via nvcuda.dll/libcuda.so at runtime
//   - Vulkan: loaded via vulkan-1.dll/libvulkan.so at runtime
//   - Metal:  loaded via Metal.framework (macOS/iOS only)
//   - SYCL:   loaded via sycl.dll/libsycl.so at runtime

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace mecan {
namespace hal {

    // ─── Universal Device Classification ────────────────────────────────────
    enum class DeviceClass {
        CPU,            // Any general-purpose processor
        GPU,            // Any graphics/compute GPU
        NPU,            // Neural Processing Unit (fixed-function AI accelerator)
        APU,            // Accelerated Processing Unit (CPU+GPU fused)
        FPGA,           // Field Programmable Gate Array
        TPU,            // Tensor Processing Unit (Google-style systolic array)
        DSP,            // Digital Signal Processor (Qualcomm Hexagon, TI C66x)
        CUSTOM,         // Any custom accelerator (ASIC, photonic, base)
        SSD_INFUSION,   // Virtual device: memory-mapped NVMe storage
    };

    // ─── Backend API type ───────────────────────────────────────────────────
    enum class BackendAPI {
        NATIVE_CPU,     // Direct C++/intrinsics (x86 AVX2, ARM NEON, etc.)
        OPENCL,         // Universal GPU/FPGA/NPU (works on ALL vendors)
        CUDA,           // NVIDIA only
        ROCM_HIP,       // AMD GPUs only
        VULKAN_COMPUTE, // Universal GPU (works on ALL vendors, even mobile)
        METAL,          // Apple Silicon only
        ONEAPI_SYCL,    // Intel GPUs, CPUs, FPGAs
        ONNXRUNTIME,    // Delegate to ONNX Runtime (supports 30+ backends)
        WEBGPU,         // Browser-based compute (Chrome, Firefox, Safari)
    };

    // ─── Device Capabilities ────────────────────────────────────────────────
    struct HWDeviceInfo {
        // Identity
        std::string name;            // "NVIDIA RTX 4090", "Apple M4 Pro", etc.
        std::string vendor;          // "NVIDIA", "AMD", "Intel", "Apple", "Qualcomm"
        std::string driver_version;
        DeviceClass device_class;
        BackendAPI  backend;

        // Compute
        uint64_t compute_units;      // SM count (GPU), core count (CPU)
        uint64_t clock_mhz;          // Max clock frequency
        double   fp32_tflops;        // Peak FP32 throughput
        double   fp16_tflops;        // Peak FP16 throughput
        double   int8_tops;          // Peak INT8 throughput
        bool     supports_fp64;
        bool     supports_fp16;
        bool     supports_bf16;
        bool     supports_int8;
        bool     supports_int4;      // Sub-byte quantization
        bool     supports_binary;    // 1-bit XNOR (our QSBits)
        bool     supports_midbits;   // 0.75-bit Block-Palette LUT Engine

        // Memory
        uint64_t global_mem_bytes;   // Total device memory
        uint64_t local_mem_bytes;    // Per-workgroup shared memory
        uint64_t max_alloc_bytes;    // Max single allocation
        double   mem_bandwidth_gbps; // Peak memory bandwidth
        bool     unified_memory;     // CPU and device share memory (APU, Apple Silicon)

        // Architecture
        uint32_t max_workgroup_size; // Max threads per workgroup
        uint32_t warp_size;          // 32 (NVIDIA), 64 (AMD), varies (Intel)
        uint32_t simd_width;         // AVX2=256bit, AVX512=512bit, NEON=128bit
        std::string isa;             // "AVX2", "NEON", "SM_89", "RDNA3", etc.
    };

    // ─── Device Memory Handle ───────────────────────────────────────────────
    struct DeviceBuffer {
        void*       host_ptr;        // Host-accessible pointer (if unified memory)
        uint64_t    device_handle;   // Opaque device memory handle
        size_t      size_bytes;
        DeviceClass device;
        int         device_index;
        bool        is_unified;      // True if host_ptr == device memory
    };

    // ─── Backend Interface ──────────────────────────────────────────────────
    // Every hardware backend implements this interface.
    // The dispatch system calls these functions based on where the tensor lives.
    class Backend {
    public:
        virtual ~Backend() = default;

        // Discovery
        virtual std::string name() const = 0;
        virtual BackendAPI  api() const = 0;
        virtual bool        is_available() const = 0;
        virtual std::vector<HWDeviceInfo> enumerate_devices() const = 0;

        // Memory
        virtual DeviceBuffer allocate(size_t nbytes, int device_index) = 0;
        virtual void         deallocate(DeviceBuffer& buf) = 0;
        virtual void         copy_h2d(const void* host, DeviceBuffer& dst, size_t nbytes) = 0;
        virtual void         copy_d2h(const DeviceBuffer& src, void* host, size_t nbytes) = 0;
        virtual void         copy_d2d(const DeviceBuffer& src, DeviceBuffer& dst, size_t nbytes) = 0;

        // Synchronization
        virtual void         synchronize(int device_index) = 0;

        // Core Operations — every backend must implement these
        virtual void matmul(const DeviceBuffer& A, const DeviceBuffer& B, DeviceBuffer& C,
                           size_t M, size_t K, size_t N) = 0;
        virtual void add(const DeviceBuffer& A, const DeviceBuffer& B, DeviceBuffer& C,
                        size_t numel) = 0;
        virtual void relu(DeviceBuffer& data, size_t numel) = 0;
        virtual void conv2d(const DeviceBuffer& input, const DeviceBuffer& filter,
                           DeviceBuffer& output,
                           size_t N, size_t C_in, size_t H, size_t W,
                           size_t C_out, size_t kH, size_t kW,
                           int stride, int pad) = 0;
        virtual void qsbits_xnor(const DeviceBuffer& input, const DeviceBuffer& weights,
                                 DeviceBuffer& output,
                                 size_t N, size_t K_packed) = 0;
        virtual void midbits_matvec(const DeviceBuffer& weights, const DeviceBuffer& X,
                                    DeviceBuffer& Y, size_t M, size_t K) = 0;
    };

    // ─── Backend Registry ───────────────────────────────────────────────────
    // Singleton that discovers and manages all available backends at runtime.
    class BackendRegistry {
    public:
        static BackendRegistry& instance();

        void register_backend(std::shared_ptr<Backend> backend);
        Backend* get_backend(BackendAPI api) const;
        Backend* get_best_backend() const;  // Auto-select fastest available
        std::vector<Backend*> list_available() const;

        // Convenience
        void discover_all();   // Probe system for all available hardware
        void print_devices();  // Print all discovered devices

    private:
        BackendRegistry() = default;
        std::vector<std::shared_ptr<Backend>> backends_;
    };

    // ─── Device Selection Helpers ───────────────────────────────────────────
    inline const char* device_class_name(DeviceClass c) {
        switch (c) {
            case DeviceClass::CPU:          return "CPU";
            case DeviceClass::GPU:          return "GPU";
            case DeviceClass::NPU:          return "NPU";
            case DeviceClass::APU:          return "APU";
            case DeviceClass::FPGA:         return "FPGA";
            case DeviceClass::TPU:          return "TPU";
            case DeviceClass::DSP:          return "DSP";
            case DeviceClass::CUSTOM:       return "CUSTOM";
            case DeviceClass::SSD_INFUSION: return "SSD";
            default: return "UNKNOWN";
        }
    }

    inline const char* backend_api_name(BackendAPI api) {
        switch (api) {
            case BackendAPI::NATIVE_CPU:     return "Native CPU";
            case BackendAPI::OPENCL:         return "OpenCL";
            case BackendAPI::CUDA:           return "CUDA";
            case BackendAPI::ROCM_HIP:       return "ROCm/HIP";
            case BackendAPI::VULKAN_COMPUTE: return "Vulkan Compute";
            case BackendAPI::METAL:          return "Metal";
            case BackendAPI::ONEAPI_SYCL:    return "oneAPI/SYCL";
            case BackendAPI::ONNXRUNTIME:    return "ONNX Runtime";
            case BackendAPI::WEBGPU:         return "WebGPU";
            default: return "Unknown";
        }
    }

} // namespace hal
} // namespace mecan
