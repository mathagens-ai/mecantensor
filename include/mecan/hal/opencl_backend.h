#pragma once
// OpenCL Backend — Universal GPU/NPU/FPGA Compute
// OpenCL runs on:
//   NVIDIA GPUs, AMD GPUs, Intel GPUs, Intel FPGAs, Qualcomm Adreno,
//   ARM Mali, Samsung NPU, Xilinx FPGAs, PowerVR GPUs, Vivante GPUs
//
// Design: Runtime dynamic loading. No OpenCL SDK needed at compile time.
//   On Windows: LoadLibrary("OpenCL.dll")
//   On Linux:   dlopen("libOpenCL.so")
//   On macOS:   dlopen("/System/Library/Frameworks/OpenCL.framework/OpenCL")

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

    // OpenCL type aliases (copied from CL/cl.h to avoid SDK dependency)
    typedef int32_t  cl_int;
    typedef uint32_t cl_uint;
    typedef uint64_t cl_ulong;
    typedef void*    cl_platform_id;
    typedef void*    cl_device_id;
    typedef void*    cl_context;
    typedef void*    cl_command_queue;
    typedef void*    cl_program;
    typedef void*    cl_kernel;
    typedef void*    cl_mem;

    // OpenCL constants
    #define MECAN_CL_DEVICE_TYPE_ALL        0xFFFFFFFF
    #define MECAN_CL_DEVICE_TYPE_GPU        (1 << 2)
    #define MECAN_CL_DEVICE_TYPE_ACCELERATOR (1 << 3)
    #define MECAN_CL_DEVICE_NAME            0x102B
    #define MECAN_CL_DEVICE_VENDOR          0x102C
    #define MECAN_CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
    #define MECAN_CL_DEVICE_MAX_CLOCK_FREQUENCY 0x100C
    #define MECAN_CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
    #define MECAN_CL_DEVICE_LOCAL_MEM_SIZE  0x1023
    #define MECAN_CL_MEM_READ_WRITE        (1 << 0)

    class OpenCLBackend : public Backend {
    public:
        OpenCLBackend() : lib_(nullptr), available_(false) {
            probe();
        }

        std::string name() const override { return "MecanTensor OpenCL (Universal GPU)"; }
        BackendAPI  api() const override { return BackendAPI::OPENCL; }
        bool        is_available() const override { return available_; }

        std::vector<HWDeviceInfo> enumerate_devices() const override {
            return discovered_devices_;
        }

        // Memory ops (stubs — filled in when OpenCL is actually loaded)
        DeviceBuffer allocate(size_t nbytes, int device_index) override {
            DeviceBuffer buf = {};
            buf.size_bytes = nbytes;
            buf.device = DeviceClass::GPU;
            buf.device_index = device_index;
            buf.is_unified = false;
            // TODO: clCreateBuffer when OpenCL context is active
            return buf;
        }
        void deallocate(DeviceBuffer& /*buf*/) override {}
        void copy_h2d(const void* /*host*/, DeviceBuffer& /*dst*/, size_t /*nbytes*/) override {}
        void copy_d2h(const DeviceBuffer& /*src*/, void* /*host*/, size_t /*nbytes*/) override {}
        void copy_d2d(const DeviceBuffer& /*src*/, DeviceBuffer& /*dst*/, size_t /*nbytes*/) override {}
        void synchronize(int /*device_index*/) override {}

        // Compute ops (stubs — implement with OpenCL kernels)
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
            // Try to load OpenCL runtime
#ifdef _WIN32
            lib_ = MECAN_LOAD_LIB("OpenCL.dll");
#elif defined(__APPLE__)
            lib_ = MECAN_LOAD_LIB("/System/Library/Frameworks/OpenCL.framework/OpenCL");
#else
            lib_ = MECAN_LOAD_LIB("libOpenCL.so.1");
            if (!lib_) lib_ = MECAN_LOAD_LIB("libOpenCL.so");
#endif
            if (!lib_) {
                available_ = false;
                return;
            }

            // Load clGetPlatformIDs
            typedef cl_int (*fn_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
            auto clGetPlatformIDs = (fn_clGetPlatformIDs)MECAN_GET_PROC(lib_, "clGetPlatformIDs");
            if (!clGetPlatformIDs) { available_ = false; return; }

            typedef cl_int (*fn_clGetDeviceIDs)(cl_platform_id, cl_ulong, cl_uint, cl_device_id*, cl_uint*);
            auto clGetDeviceIDs = (fn_clGetDeviceIDs)MECAN_GET_PROC(lib_, "clGetDeviceIDs");

            typedef cl_int (*fn_clGetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*);
            auto clGetDeviceInfo = (fn_clGetDeviceInfo)MECAN_GET_PROC(lib_, "clGetDeviceInfo");

            if (!clGetDeviceIDs || !clGetDeviceInfo) { available_ = false; return; }

            // Enumerate platforms
            cl_uint num_platforms = 0;
            clGetPlatformIDs(0, nullptr, &num_platforms);
            if (num_platforms == 0) { available_ = false; return; }

            std::vector<cl_platform_id> platforms(num_platforms);
            clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

            // Enumerate devices on each platform
            int dev_idx = 0;
            for (cl_uint p = 0; p < num_platforms; p++) {
                cl_uint num_devices = 0;
                clGetDeviceIDs(platforms[p], MECAN_CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
                if (num_devices == 0) continue;

                std::vector<cl_device_id> devices(num_devices);
                clGetDeviceIDs(platforms[p], MECAN_CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);

                for (cl_uint d = 0; d < num_devices; d++) {
                    HWDeviceInfo cap = {};
                    char buf[256] = {};

                    clGetDeviceInfo(devices[d], MECAN_CL_DEVICE_NAME, sizeof(buf), buf, nullptr);
                    cap.name = buf;

                    clGetDeviceInfo(devices[d], MECAN_CL_DEVICE_VENDOR, sizeof(buf), buf, nullptr);
                    cap.vendor = buf;

                    cl_uint cu = 0;
                    clGetDeviceInfo(devices[d], MECAN_CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
                    cap.compute_units = cu;

                    cl_uint freq = 0;
                    clGetDeviceInfo(devices[d], MECAN_CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(freq), &freq, nullptr);
                    cap.clock_mhz = freq;

                    cl_ulong gmem = 0;
                    clGetDeviceInfo(devices[d], MECAN_CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, nullptr);
                    cap.global_mem_bytes = gmem;

                    cl_ulong lmem = 0;
                    clGetDeviceInfo(devices[d], MECAN_CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, nullptr);
                    cap.local_mem_bytes = lmem;

                    cap.device_class = DeviceClass::GPU;
                    cap.backend = BackendAPI::OPENCL;
                    cap.supports_fp64 = false;
                    cap.supports_fp16 = true;
                    cap.supports_int8 = true;
                    cap.supports_binary = true;
                    cap.supports_midbits = true; // OpenCL can do LUTs very fast!
                    cap.driver_version = "OpenCL Runtime";

                    discovered_devices_.push_back(cap);
                    dev_idx++;
                }
            }

            available_ = !discovered_devices_.empty();
        }
    };

} // namespace hal
} // namespace mecan
