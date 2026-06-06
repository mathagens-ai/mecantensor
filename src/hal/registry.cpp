// Backend Registry + Hardware Discovery Engine
// At startup, probes the system for ALL available compute devices:
//   1. CPU (always available)
//   2. OpenCL (NVIDIA/AMD/Intel/Qualcomm/ARM GPUs, FPGAs, NPUs)
//   3. CUDA (NVIDIA GPUs only, via nvcuda.dll)
//   4. Vulkan (any modern GPU, via vulkan-1.dll)
//   5. Metal (Apple Silicon, via Metal.framework)
//   6. SYCL (Intel GPUs/CPUs/FPGAs, via sycl.dll)

#include "mecan/hal/hal.h"
#include "mecan/hal/cpu_backend.h"
#include "mecan/hal/opencl_backend.h"
#include "mecan/hal/cuda_backend.h"
#include "mecan/hal/vulkan_backend.h"
#include <cstdio>
#include <algorithm>

namespace mecan {
namespace hal {

    // ─── Backend Registry Implementation ────────────────────────────────────
    BackendRegistry& BackendRegistry::instance() {
        static BackendRegistry reg;
        return reg;
    }

    void BackendRegistry::register_backend(std::shared_ptr<Backend> backend) {
        backends_.push_back(backend);
    }

    Backend* BackendRegistry::get_backend(BackendAPI api) const {
        for (auto& b : backends_) {
            if (b->api() == api && b->is_available()) return b.get();
        }
        return nullptr;
    }

    Backend* BackendRegistry::get_best_backend() const {
        // Priority: CUDA > ROCm > Vulkan > OpenCL > SYCL > CPU
        static const BackendAPI priority[] = {
            BackendAPI::CUDA,
            BackendAPI::ROCM_HIP,
            BackendAPI::VULKAN_COMPUTE,
            BackendAPI::OPENCL,
            BackendAPI::ONEAPI_SYCL,
            BackendAPI::METAL,
            BackendAPI::NATIVE_CPU,
        };

        for (auto api : priority) {
            Backend* b = get_backend(api);
            if (b && b->is_available()) return b;
        }
        return nullptr;
    }

    std::vector<Backend*> BackendRegistry::list_available() const {
        std::vector<Backend*> avail;
        for (auto& b : backends_) {
            if (b->is_available()) avail.push_back(b.get());
        }
        return avail;
    }

    void BackendRegistry::discover_all() {
        // Always register CPU first
        register_backend(std::make_shared<CPUBackend>());

        // Probe CUDA backend
        auto cuda = std::make_shared<CUDABackend>();
        register_backend(cuda);

        // Probe Vulkan backend
        auto vulkan = std::make_shared<VulkanBackend>();
        register_backend(vulkan);

        // Probe OpenCL backend
        auto opencl = std::make_shared<OpenCLBackend>();
        register_backend(opencl);
    }

    void BackendRegistry::print_devices() {
        printf(" MECANTENSOR: UNIVERSAL HARDWARE DISCOVERY\n");

        int total_devices = 0;
        for (auto& b : backends_) {
            bool avail = b->is_available();
            printf("  Backend: %-25s [%s]\n", b->name().c_str(), avail ? "AVAILABLE" : "not found");

            if (avail) {
                auto devices = b->enumerate_devices();
                for (size_t i = 0; i < devices.size(); i++) {
                    auto& d = devices[i];
                    printf("    [%d] %-40s (%s)\n", (int)i, d.name.c_str(), d.vendor.c_str());
                    printf("        Class: %-8s | Backend: %s\n",
                           device_class_name(d.device_class), backend_api_name(d.backend));
                    printf("        Compute Units: %llu | Clock: %llu MHz\n",
                           (unsigned long long)d.compute_units, (unsigned long long)d.clock_mhz);

                    if (d.global_mem_bytes > 0) {
                        printf("        Memory: %.1f GB (%.1f GB/s bandwidth)\n",
                               d.global_mem_bytes / (1024.0*1024.0*1024.0),
                               d.mem_bandwidth_gbps);
                    }

                    printf("        SIMD: %u-bit (%s)\n", d.simd_width, d.isa.c_str());
                    printf("        Capabilities: FP32");
                    if (d.supports_fp64) printf(" FP64");
                    if (d.supports_fp16) printf(" FP16");
                    if (d.supports_bf16) printf(" BF16");
                    if (d.supports_int8) printf(" INT8");
                    if (d.supports_int4) printf(" INT4");
                    if (d.supports_binary) printf(" BINARY(QSBits)");
                    if (d.supports_midbits) printf(" MIDBITS(0.75b)");
                    printf("\n");

                    if (d.fp32_tflops > 0) {
                        printf("        Peak FP32: %.3f TFLOPS\n", d.fp32_tflops);
                    }

                    printf("\n");
                    total_devices++;
                }
            }
        }

        printf("  Total devices discovered: %d\n", total_devices);

        Backend* best = get_best_backend();
        if (best) {
            printf("  Auto-selected backend: %s\n", best->name().c_str());
        }

    }

} // namespace hal
} // namespace mecan
