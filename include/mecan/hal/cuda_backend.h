#pragma once

#include "mecan/hal/hal.h"

#ifdef _WIN32
    #include <windows.h>
    #define MECAN_LOAD_LIB(x)    LoadLibraryA(x)
    #define MECAN_GET_PROC(h,n)  GetProcAddress((HMODULE)(h), n)
    #define MECAN_LIB_HANDLE     HMODULE
#else
    #include <dlfcn.h>
    #define MECAN_LOAD_LIB(x)    dlopen(x, RTLD_LAZY)
    #define MECAN_GET_PROC(h,n)  dlsym(h, n)
    #define MECAN_LIB_HANDLE     void*
#endif

namespace mecan {
namespace hal {

    typedef int CUdevice;
    typedef int CUresult;

    #define CUDA_SUCCESS 0
    #define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16
    #define CU_DEVICE_ATTRIBUTE_CLOCK_RATE 13
    #define CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING 41
    #define CU_DEVICE_ATTRIBUTE_WARP_SIZE 10

    class CUDABackend : public Backend {
    public:
        CUDABackend() : lib_(nullptr), available_(false) {
            probe();
        }

        ~CUDABackend() override {
            // Library handle released at shutdown
        }

        std::string name() const override { return "MecanTensor CUDA Backend"; }
        BackendAPI  api() const override { return BackendAPI::CUDA; }
        bool        is_available() const override { return available_; }

        std::vector<HWDeviceInfo> enumerate_devices() const override {
            return discovered_devices_;
        }

        DeviceBuffer allocate(size_t nbytes, int device_index) override {
            DeviceBuffer buf = {};
            buf.size_bytes = nbytes;
            buf.device = DeviceClass::GPU;
            buf.device_index = device_index;
            buf.is_unified = false;
            return buf;
        }

        void deallocate(DeviceBuffer& /*buf*/) override {}
        void copy_h2d(const void* /*host*/, DeviceBuffer& /*dst*/, size_t /*nbytes*/) override {}
        void copy_d2h(const DeviceBuffer& /*src*/, void* /*host*/, size_t /*nbytes*/) override {}
        void copy_d2d(const DeviceBuffer& /*src*/, DeviceBuffer& /*dst*/, size_t /*nbytes*/) override {}
        void synchronize(int /*device_index*/) override {}

        void matmul(const DeviceBuffer&, const DeviceBuffer&, DeviceBuffer&,
                   size_t, size_t, size_t) override {}
        void add(const DeviceBuffer&, const DeviceBuffer&, DeviceBuffer&,
                size_t) override {}
        void relu(DeviceBuffer&, size_t) override {}
        void conv2d(const DeviceBuffer&, const DeviceBuffer&, DeviceBuffer&,
                   size_t, size_t, size_t, size_t, size_t, size_t, size_t,
                   int, int) override {}
        void qsbits_xnor(const DeviceBuffer&, const DeviceBuffer&, DeviceBuffer&,
                         size_t, size_t) override {}
        void midbits_matvec(const DeviceBuffer&, const DeviceBuffer&, DeviceBuffer&,
                            size_t, size_t) override {}

    private:
        MECAN_LIB_HANDLE lib_;
        bool available_;
        std::vector<HWDeviceInfo> discovered_devices_;

        void probe() {
#ifdef _WIN32
            lib_ = MECAN_LOAD_LIB("nvcuda.dll");
#else
            lib_ = MECAN_LOAD_LIB("libcuda.so.1");
            if (!lib_) lib_ = MECAN_LOAD_LIB("libcuda.so");
#endif
            if (!lib_) {
                available_ = false;
                return;
            }

            typedef CUresult (*fn_cuInit)(unsigned int);
            typedef CUresult (*fn_cuDeviceGetCount)(int*);
            typedef CUresult (*fn_cuDeviceGet)(CUdevice*, int);
            typedef CUresult (*fn_cuDeviceGetName)(char*, int, CUdevice);
            typedef CUresult (*fn_cuDeviceTotalMem)(size_t*, CUdevice);
            typedef CUresult (*fn_cuDeviceGetAttribute)(int*, int, CUdevice);

            auto cuInit = (fn_cuInit)MECAN_GET_PROC(lib_, "cuInit");
            auto cuDeviceGetCount = (fn_cuDeviceGetCount)MECAN_GET_PROC(lib_, "cuDeviceGetCount");
            auto cuDeviceGet = (fn_cuDeviceGet)MECAN_GET_PROC(lib_, "cuDeviceGet");
            auto cuDeviceGetName = (fn_cuDeviceGetName)MECAN_GET_PROC(lib_, "cuDeviceGetName");
            auto cuDeviceTotalMem = (fn_cuDeviceTotalMem)MECAN_GET_PROC(lib_, "cuDeviceTotalMem");
            auto cuDeviceGetAttribute = (fn_cuDeviceGetAttribute)MECAN_GET_PROC(lib_, "cuDeviceGetAttribute");

            if (!cuInit || !cuDeviceGetCount || !cuDeviceGet || !cuDeviceGetName || !cuDeviceTotalMem || !cuDeviceGetAttribute) {
                available_ = false;
                return;
            }

            if (cuInit(0) != CUDA_SUCCESS) {
                available_ = false;
                return;
            }

            int device_count = 0;
            if (cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count == 0) {
                available_ = false;
                return;
            }

            for (int i = 0; i < device_count; ++i) {
                CUdevice dev;
                if (cuDeviceGet(&dev, i) != CUDA_SUCCESS) continue;

                HWDeviceInfo cap = {};
                char name_buf[256] = {};
                if (cuDeviceGetName(name_buf, sizeof(name_buf), dev) == CUDA_SUCCESS) {
                    cap.name = name_buf;
                } else {
                    cap.name = "NVIDIA CUDA Device";
                }

                cap.vendor = "NVIDIA";
                cap.driver_version = "CUDA Driver API";
                cap.device_class = DeviceClass::GPU;
                cap.backend = BackendAPI::CUDA;

                size_t total_mem = 0;
                if (cuDeviceTotalMem(&total_mem, dev) == CUDA_SUCCESS) {
                    cap.global_mem_bytes = total_mem;
                }

                int mp_count = 0;
                if (cuDeviceGetAttribute(&mp_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev) == CUDA_SUCCESS) {
                    cap.compute_units = mp_count;
                }

                int clock_rate = 0;
                if (cuDeviceGetAttribute(&clock_rate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev) == CUDA_SUCCESS) {
                    cap.clock_mhz = clock_rate / 1000; // Kilohertz to megahertz
                }

                int warp_size = 0;
                if (cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev) == CUDA_SUCCESS) {
                    cap.warp_size = warp_size;
                }

                int unified_addr = 0;
                if (cuDeviceGetAttribute(&unified_addr, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, dev) == CUDA_SUCCESS) {
                    cap.unified_memory = (unified_addr != 0);
                }

                cap.supports_fp64 = true;
                cap.supports_fp16 = true;
                cap.supports_bf16 = true;
                cap.supports_int8 = true;
                cap.supports_binary = true;

                discovered_devices_.push_back(cap);
            }

            available_ = !discovered_devices_.empty();
        }
    };

} // namespace hal
} // namespace mecan
