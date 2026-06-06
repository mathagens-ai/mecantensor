#pragma once
// CPU Backend — Native x86/ARM/RISC-V Implementation
// This is the "always available" backend. Every machine has a CPU.
// Uses: OpenBLAS (matmul), AVX2 intrinsics (QSBits, add), OpenMP (threading)

#include "mecan/hal/hal.h"
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>

#ifdef _WIN32
    #include <windows.h>
    #include <malloc.h>
    #include <intrin.h>
#else
    #include <mm_malloc.h>
#endif

namespace mecan {
namespace hal {

    class CPUBackend : public Backend {
    public:
        std::string name() const override { return "MecanTensor Native CPU"; }
        BackendAPI  api() const override { return BackendAPI::NATIVE_CPU; }
        bool        is_available() const override { return true; } // CPU is always available

        std::vector<HWDeviceInfo> enumerate_devices() const override {
            HWDeviceInfo cap = {};
            cap.name = detect_cpu_name();
            cap.vendor = detect_cpu_vendor();
            cap.driver_version = "native";
            cap.device_class = DeviceClass::CPU;
            cap.backend = BackendAPI::NATIVE_CPU;

            cap.compute_units = detect_core_count();
            cap.clock_mhz = detect_clock_mhz();
            cap.supports_fp64 = true;
            cap.supports_fp16 = false; // Software emulation only
            cap.supports_bf16 = detect_bf16_support();
            cap.supports_int8 = true;
            cap.supports_int4 = false;
            cap.supports_binary = true; // QSBits XNOR!
            cap.supports_midbits = true; // MidBits LUT AVX2 Engine!
            cap.unified_memory = true;  // CPU memory IS host memory

            cap.simd_width = detect_simd_width();
            cap.isa = detect_isa();

            // Estimate peak TFLOPS based on cores × clock × FMA width
            double ghz = cap.clock_mhz / 1000.0;
            int fma_width = cap.simd_width / 32; // floats per SIMD register
            cap.fp32_tflops = cap.compute_units * ghz * fma_width * 2.0 / 1000.0; // ×2 for FMA

            return {cap};
        }

        // Memory
        DeviceBuffer allocate(size_t nbytes, int /*device_index*/) override {
            DeviceBuffer buf = {};
            buf.size_bytes = nbytes;
            buf.device = DeviceClass::CPU;
            buf.device_index = 0;
            buf.is_unified = true;
#ifdef _WIN32
            buf.host_ptr = _aligned_malloc(nbytes, 64);
#else
            buf.host_ptr = nullptr;
            posix_memalign(&buf.host_ptr, 64, nbytes);
#endif
            buf.device_handle = (uint64_t)buf.host_ptr;
            return buf;
        }

        void deallocate(DeviceBuffer& buf) override {
            if (buf.host_ptr) {
#ifdef _WIN32
                _aligned_free(buf.host_ptr);
#else
                free(buf.host_ptr);
#endif
                buf.host_ptr = nullptr;
            }
        }

        void copy_h2d(const void* host, DeviceBuffer& dst, size_t nbytes) override {
            memcpy(dst.host_ptr, host, nbytes); // CPU: h2d is just memcpy
        }
        void copy_d2h(const DeviceBuffer& src, void* host, size_t nbytes) override {
            memcpy(host, src.host_ptr, nbytes);
        }
        void copy_d2d(const DeviceBuffer& src, DeviceBuffer& dst, size_t nbytes) override {
            memcpy(dst.host_ptr, src.host_ptr, nbytes);
        }

        void synchronize(int /*device_index*/) override {
            // CPU operations are synchronous — nothing to wait for
        }

        // Core ops delegate to the existing optimized kernels (stubbed for discovery)
        void matmul(const DeviceBuffer& A, const DeviceBuffer& B, DeviceBuffer& C,
                   size_t M, size_t K, size_t N) override {}
        void add(const DeviceBuffer& A, const DeviceBuffer& B, DeviceBuffer& C,
                size_t numel) override {}
        void relu(DeviceBuffer& data, size_t numel) override {}
        void conv2d(const DeviceBuffer& input, const DeviceBuffer& filter,
                   DeviceBuffer& output,
                   size_t N, size_t C_in, size_t H, size_t W,
                   size_t C_out, size_t kH, size_t kW,
                   int stride, int pad) override {}
        void qsbits_xnor(const DeviceBuffer& input, const DeviceBuffer& weights,
                         DeviceBuffer& output,
                         size_t N, size_t K_packed) override {}
        void midbits_matvec(const DeviceBuffer& weights, const DeviceBuffer& X,
                            DeviceBuffer& Y, size_t M, size_t K) override {}

    private:
        // ─── CPU Detection ──────────────────────────────────────────────
        static std::string detect_cpu_name() {
#if defined(_MSC_VER)
            int info[4] = {};
            char brand[49] = {};
            __cpuid(info, 0x80000002); memcpy(brand, info, 16);
            __cpuid(info, 0x80000003); memcpy(brand+16, info, 16);
            __cpuid(info, 0x80000004); memcpy(brand+32, info, 16);
            brand[48] = 0;
            // Trim leading spaces
            char* p = brand;
            while (*p == ' ') p++;
            return std::string(p);
#else
            return "CPU (use /proc/cpuinfo)";
#endif
        }

        static std::string detect_cpu_vendor() {
#if defined(_MSC_VER)
            int info[4] = {};
            __cpuid(info, 0);
            char vendor[13] = {};
            memcpy(vendor, &info[1], 4);
            memcpy(vendor+4, &info[3], 4);
            memcpy(vendor+8, &info[2], 4);
            vendor[12] = 0;
            if (strstr(vendor, "Intel")) return "Intel";
            if (strstr(vendor, "AMD")) return "AMD";
            return vendor;
#else
            return "Unknown";
#endif
        }

        static uint32_t detect_core_count() {
#ifdef _WIN32
            SYSTEM_INFO si;
            GetSystemInfo(&si);
            return si.dwNumberOfProcessors;
#else
            return 1;
#endif
        }

        static uint32_t detect_clock_mhz() {
            return 3500; // Default, can be refined with CPUID leaf 0x16
        }

        static uint32_t detect_simd_width() {
#if defined(_MSC_VER)
            int info[4] = {};
            __cpuidex(info, 7, 0);
            if (info[1] & (1 << 16)) return 512; // AVX-512
            if (info[1] & (1 << 5))  return 256; // AVX2
            __cpuid(info, 1);
            if (info[2] & (1 << 28)) return 256; // AVX
            return 128; // SSE
#elif defined(__ARM_NEON)
            return 128; // NEON
#else
            return 64;  // Scalar fallback
#endif
        }

        static std::string detect_isa() {
#if defined(_MSC_VER)
            int info[4] = {};
            __cpuidex(info, 7, 0);
            if (info[1] & (1 << 16)) return "AVX-512";
            if (info[1] & (1 << 5))  return "AVX2";
            __cpuid(info, 1);
            if (info[2] & (1 << 28)) return "AVX";
            return "SSE2";
#elif defined(__ARM_NEON)
            return "NEON";
#elif defined(__riscv)
            return "RISC-V";
#else
            return "Scalar";
#endif
        }

        static bool detect_bf16_support() {
#if defined(_MSC_VER)
            int info[4] = {};
            __cpuidex(info, 7, 1);
            return (info[0] & (1 << 5)) != 0; // AVX-512 BF16
#else
            return false;
#endif
        }
    };

} // namespace hal
} // namespace mecan
