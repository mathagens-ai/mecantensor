// mecantensor/src/hlas/gpu_engine.cpp
#include "hlas.h"
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace mecan {
namespace hlas {

// ─── OPENCL CONSTANTS & TYPES ───────────────────────────────────────────────
typedef int cl_int;
typedef unsigned int cl_uint;
typedef unsigned long long cl_ulong;
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_mem;
typedef void* cl_program;
typedef void* cl_kernel;
typedef void* cl_event;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define CL_PROGRAM_BUILD_LOG 0x1183
#define CL_PROGRAM_BUILD_FAILURE -11
#define CL_MEM_READ_ONLY (1 << 2)
#define CL_MEM_WRITE_ONLY (1 << 1)
#define CL_MEM_COPY_HOST_PTR (1 << 5)

// ─── DYNAMIC OPENCL FUNCTION POINTERS ───────────────────────────────────────
typedef cl_int (*PFN_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*PFN_clGetDeviceIDs)(cl_platform_id, cl_ulong, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (*PFN_clGetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_context (*PFN_clCreateContext)(const void*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
typedef cl_command_queue (*PFN_clCreateCommandQueue)(cl_context, cl_device_id, cl_ulong, cl_int*);
typedef cl_program (*PFN_clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (*PFN_clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
typedef cl_int (*PFN_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_kernel (*PFN_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_mem (*PFN_clCreateBuffer)(cl_context, cl_ulong, size_t, void*, cl_int*);
typedef cl_int (*PFN_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (*PFN_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*PFN_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*PFN_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*PFN_clFinish)(cl_command_queue);
typedef cl_int (*PFN_clReleaseMemObject)(cl_mem);
typedef cl_int (*PFN_clReleaseKernel)(cl_kernel);
typedef cl_int (*PFN_clReleaseProgram)(cl_program);
typedef cl_int (*PFN_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int (*PFN_clReleaseContext)(cl_context);

struct OpenCLFunctions {
    PFN_clGetPlatformIDs clGetPlatformIDs = nullptr;
    PFN_clGetDeviceIDs clGetDeviceIDs = nullptr;
    PFN_clGetDeviceInfo clGetDeviceInfo = nullptr;
    PFN_clCreateContext clCreateContext = nullptr;
    PFN_clCreateCommandQueue clCreateCommandQueue = nullptr;
    PFN_clCreateProgramWithSource clCreateProgramWithSource = nullptr;
    PFN_clBuildProgram clBuildProgram = nullptr;
    PFN_clGetProgramBuildInfo clGetProgramBuildInfo = nullptr;
    PFN_clCreateKernel clCreateKernel = nullptr;
    PFN_clCreateBuffer clCreateBuffer = nullptr;
    PFN_clSetKernelArg clSetKernelArg = nullptr;
    PFN_clEnqueueWriteBuffer clEnqueueWriteBuffer = nullptr;
    PFN_clEnqueueReadBuffer clEnqueueReadBuffer = nullptr;
    PFN_clEnqueueNDRangeKernel clEnqueueNDRangeKernel = nullptr;
    PFN_clFinish clFinish = nullptr;
    PFN_clReleaseMemObject clReleaseMemObject = nullptr;
    PFN_clReleaseKernel clReleaseKernel = nullptr;
    PFN_clReleaseProgram clReleaseProgram = nullptr;
    PFN_clReleaseCommandQueue clReleaseCommandQueue = nullptr;
    PFN_clReleaseContext clReleaseContext = nullptr;
};

static OpenCLFunctions cl;
static bool ocl_loaded = false;

// ─── NATIVE OPENCL KERNEL SOURCE ───────────────────────────────────────────
const char* GEMM_KERNEL_SRC = R"(
// Universal Tiled GEMM Kernel
#define TILE 16

__kernel void mecan_gemm(
    const int M,
    const int N,
    const int K,
    const __global float* A,
    const __global float* B,
    __global float* C
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);

    __local float tile_a[TILE][TILE];
    __local float tile_b[TILE][TILE];

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        const int a_col = t * TILE + local_col;
        const int b_row = t * TILE + local_row;

        tile_a[local_row][local_col] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        tile_b[local_row][local_col] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k0 = 0; k0 < TILE; ++k0) {
            acc += tile_a[local_row][k0] * tile_b[k0][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// 1-Bit XNOR Kernel (QSBits)
__kernel void mecan_qsbits_xnor(
    const int M,
    const int K_packed,
    const __global uchar* input,
    const __global uchar* weights,
    __global float* output
) {
    const int row = get_global_id(0);
    if (row < M) {
        int total_pop = 0;
        const __global uchar* w_row = weights + row * K_packed;
        
        // Loop over bytes
        for (int k = 0; k < K_packed; ++k) {
            uchar xnor_val = ~(input[k] ^ w_row[k]);
            total_pop += popcount(xnor_val);
        }
        
        output[row] = (float)(2 * total_pop - K_packed * 8);
    }
}

__kernel void mecan_qsbits_xnor_scaled(
    const int M,
    const int K_packed,
    const __global uchar* input,
    const __global uchar* weights,
    const __global float* scales,
    const float input_scale,
    const int g_packed,
    const int n_groups,
    __global float* output
) {
    const int row = get_global_id(0);
    if (row >= M) return;
    float acc = 0.0f;
    for (int g = 0; g < n_groups; ++g) {
        int base = g * g_packed;
        int pop = 0;
        for (int k = 0; k < g_packed; ++k) {
            uchar xnor_val = ~(input[base + k] ^ weights[row * K_packed + base + k]);
            pop += popcount(xnor_val);
        }
        float val = (float)(2 * pop - g_packed * 8);
        acc += val * scales[row * n_groups + g];
    }
    output[row] = acc * input_scale;
}
)";

// ─── GPU ENGINE IMPLEMENTATION ─────────────────────────────────────────────

GpuEngine::GpuEngine() : initialized_(false), context_(nullptr), queue_(nullptr), program_(nullptr) {
    calibrate();
}

GpuEngine::~GpuEngine() {
    if (initialized_ && ocl_loaded) {
        // We only retain context, queue, and program at class level.
        if (program_) cl.clReleaseProgram((cl_program)program_);
        if (queue_) cl.clReleaseCommandQueue((cl_command_queue)queue_);
        if (context_) cl.clReleaseContext((cl_context)context_);
    }
}

void GpuEngine::calibrate() {
    info_.name = "No GPU Detected";
    info_.type = BackendType::GPU_OPENCL;
    info_.max_vram = 0;
    info_.compute_units = 0;

    // 1. Dynamically Load OpenCL Library
#ifdef _WIN32
    HMODULE lib = LoadLibraryA("OpenCL.dll");
#else
    void* lib = dlopen("libOpenCL.so", RTLD_LAZY);
#endif

    if (!lib) return; // Silent fallback

#ifdef _WIN32
    #define LOAD_FN(name) cl.name = (PFN_##name)GetProcAddress(lib, #name);
#else
    #define LOAD_FN(name) cl.name = (PFN_##name)dlsym(lib, #name);
#endif

    LOAD_FN(clGetPlatformIDs);
    LOAD_FN(clGetDeviceIDs);
    LOAD_FN(clGetDeviceInfo);
    LOAD_FN(clCreateContext);
    LOAD_FN(clCreateCommandQueue);
    LOAD_FN(clCreateProgramWithSource);
    LOAD_FN(clBuildProgram);
    LOAD_FN(clGetProgramBuildInfo);
    LOAD_FN(clCreateKernel);
    LOAD_FN(clCreateBuffer);
    LOAD_FN(clSetKernelArg);
    LOAD_FN(clEnqueueWriteBuffer);
    LOAD_FN(clEnqueueReadBuffer);
    LOAD_FN(clEnqueueNDRangeKernel);
    LOAD_FN(clFinish);
    LOAD_FN(clReleaseMemObject);
    LOAD_FN(clReleaseKernel);
    LOAD_FN(clReleaseProgram);
    LOAD_FN(clReleaseCommandQueue);
    LOAD_FN(clReleaseContext);

    if (!cl.clGetPlatformIDs || !cl.clCreateContext) return;
    ocl_loaded = true;

    // 2. Discover GPU
    cl_uint numPlatforms;
    if (cl.clGetPlatformIDs(0, nullptr, &numPlatforms) != CL_SUCCESS || numPlatforms == 0) return;

    std::vector<cl_platform_id> platforms(numPlatforms);
    cl.clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    cl_device_id targetDevice = nullptr;
    char deviceName[256];
    cl_uint computeUnits = 0;
    cl_ulong memSize = 0;

    for (cl_uint i = 0; i < numPlatforms; i++) {
        cl_uint numDevices;
        if (cl.clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices) == CL_SUCCESS && numDevices > 0) {
            std::vector<cl_device_id> devices(numDevices);
            cl.clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
            targetDevice = devices[0];
            
            cl.clGetDeviceInfo(targetDevice, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
            cl.clGetDeviceInfo(targetDevice, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, nullptr);
            cl.clGetDeviceInfo(targetDevice, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memSize), &memSize, nullptr);
            break;
        }
    }

    if (!targetDevice) return;

    info_.name = deviceName;
    info_.compute_units = computeUnits;
    info_.max_vram = memSize;

    // 3. Establish Context and Compile
    cl_int err;
    cl_context ctx = cl.clCreateContext(nullptr, 1, &targetDevice, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) return;

    cl_command_queue q = cl.clCreateCommandQueue(ctx, targetDevice, 0, &err);
    if (err != CL_SUCCESS) { cl.clReleaseContext(ctx); return; }

    cl_program prog = cl.clCreateProgramWithSource(ctx, 1, &GEMM_KERNEL_SRC, nullptr, &err);
    if (err != CL_SUCCESS) { cl.clReleaseCommandQueue(q); cl.clReleaseContext(ctx); return; }

    err = cl.clBuildProgram(prog, 1, &targetDevice, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        cl.clReleaseProgram(prog);
        cl.clReleaseCommandQueue(q);
        cl.clReleaseContext(ctx);
        return;
    }

    // Success
    context_ = ctx;
    queue_ = q;
    program_ = prog;
    initialized_ = true;
}

DeviceInfo GpuEngine::get_device_info() const {
    return info_;
}

void GpuEngine::sgemm(
    int M, int N, int K,
    float alpha, const float* A, int lda,
    const float* B, int ldb,
    float beta, float* C, int ldc,
    FusedActivation act) 
{
    if (!initialized_) return;

    cl_context ctx = (cl_context)context_;
    cl_command_queue q = (cl_command_queue)queue_;
    cl_program prog = (cl_program)program_;

    cl_int err;
    cl_kernel kernel = cl.clCreateKernel(prog, "mecan_gemm", &err);
    if (err != CL_SUCCESS) return;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    cl_mem devA = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeA, (void*)A, &err);
    cl_mem devB = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeB, (void*)B, &err);
    cl_mem devC = cl.clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeC, nullptr, &err);

    cl.clSetKernelArg(kernel, 0, sizeof(int), &M);
    cl.clSetKernelArg(kernel, 1, sizeof(int), &N);
    cl.clSetKernelArg(kernel, 2, sizeof(int), &K);
    cl.clSetKernelArg(kernel, 3, sizeof(cl_mem), &devA);
    cl.clSetKernelArg(kernel, 4, sizeof(cl_mem), &devB);
    cl.clSetKernelArg(kernel, 5, sizeof(cl_mem), &devC);

    const size_t tile = 16;
    size_t globalWorkSize[2] = { ((size_t)M + tile - 1) / tile * tile, ((size_t)N + tile - 1) / tile * tile };
    size_t localWorkSize[2] = { tile, tile };

    cl.clEnqueueNDRangeKernel(q, kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    cl.clEnqueueReadBuffer(q, devC, 1, 0, sizeC, C, 0, nullptr, nullptr);
    cl.clFinish(q);

    cl.clReleaseMemObject(devA);
    cl.clReleaseMemObject(devB);
    cl.clReleaseMemObject(devC);
    cl.clReleaseKernel(kernel);
}

void GpuEngine::conv2d_fused(const float*, int, int, int, const float*, int, int, int, float*, int, int, FusedActivation) {}
void GpuEngine::midbits_sgemm(int, int, int, const uint8_t*, const float*, float*, FusedActivation) {}
void GpuEngine::qsbits_xnor(int M, int K_packed, const uint8_t* input, const uint8_t* weights, float* output) {
    if (!initialized_) return;

    cl_context ctx = (cl_context)context_;
    cl_command_queue q = (cl_command_queue)queue_;
    cl_program prog = (cl_program)program_;

    cl_int err;
    cl_kernel kernel = cl.clCreateKernel(prog, "mecan_qsbits_xnor", &err);
    if (err != CL_SUCCESS) return;

    size_t size_in = K_packed * sizeof(uint8_t);
    size_t size_w = M * K_packed * sizeof(uint8_t);
    size_t size_out = M * sizeof(float);

    cl_mem dev_in = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_in, (void*)input, &err);
    cl_mem dev_w = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_w, (void*)weights, &err);
    cl_mem dev_out = cl.clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, size_out, nullptr, &err);

    cl.clSetKernelArg(kernel, 0, sizeof(int), &M);
    cl.clSetKernelArg(kernel, 1, sizeof(int), &K_packed);
    cl.clSetKernelArg(kernel, 2, sizeof(cl_mem), &dev_in);
    cl.clSetKernelArg(kernel, 3, sizeof(cl_mem), &dev_w);
    cl.clSetKernelArg(kernel, 4, sizeof(cl_mem), &dev_out);

    size_t globalWorkSize[1] = { (size_t)M };
    size_t localWorkSize[1] = { 64 };
    
    // Ensure globalWorkSize is a multiple of localWorkSize
    globalWorkSize[0] = ((globalWorkSize[0] + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];

    cl.clEnqueueNDRangeKernel(q, kernel, 1, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    cl.clEnqueueReadBuffer(q, dev_out, 1, 0, size_out, output, 0, nullptr, nullptr);
    cl.clFinish(q);

    cl.clReleaseMemObject(dev_in);
    cl.clReleaseMemObject(dev_w);
    cl.clReleaseMemObject(dev_out);
    cl.clReleaseKernel(kernel);
}

void GpuEngine::qsbits_xnor_scaled(
    int M, int K_packed, const uint8_t* input, const uint8_t* weights,
    const float* scales, float input_scale, int group_size, float* output)
{
    if (!initialized_) return;

    cl_context ctx = (cl_context)context_;
    cl_command_queue q = (cl_command_queue)queue_;
    cl_program prog = (cl_program)program_;

    cl_int err;
    cl_kernel kernel = cl.clCreateKernel(prog, "mecan_qsbits_xnor_scaled", &err);
    if (err != CL_SUCCESS) return;

    int g_packed = group_size / 8;
    int n_groups = K_packed / g_packed;

    size_t size_in = K_packed * sizeof(uint8_t);
    size_t size_w  = M * K_packed * sizeof(uint8_t);
    size_t size_sc = M * n_groups * sizeof(float);
    size_t size_out = M * sizeof(float);

    cl_mem dev_in  = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_in, (void*)input, &err);
    cl_mem dev_w   = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_w, (void*)weights, &err);
    cl_mem dev_sc  = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_sc, (void*)scales, &err);
    cl_mem dev_out = cl.clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, size_out, nullptr, &err);

    cl.clSetKernelArg(kernel, 0, sizeof(int), &M);
    cl.clSetKernelArg(kernel, 1, sizeof(int), &K_packed);
    cl.clSetKernelArg(kernel, 2, sizeof(cl_mem), &dev_in);
    cl.clSetKernelArg(kernel, 3, sizeof(cl_mem), &dev_w);
    cl.clSetKernelArg(kernel, 4, sizeof(cl_mem), &dev_sc);
    cl.clSetKernelArg(kernel, 5, sizeof(float), &input_scale);
    cl.clSetKernelArg(kernel, 6, sizeof(int), &g_packed);
    cl.clSetKernelArg(kernel, 7, sizeof(int), &n_groups);
    cl.clSetKernelArg(kernel, 8, sizeof(cl_mem), &dev_out);

    size_t globalWorkSize[1] = { ((size_t)M + 63) / 64 * 64 };
    size_t localWorkSize[1] = { 64 };

    cl.clEnqueueNDRangeKernel(q, kernel, 1, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    cl.clEnqueueReadBuffer(q, dev_out, 1, 0, size_out, output, 0, nullptr, nullptr);
    cl.clFinish(q);

    cl.clReleaseMemObject(dev_in);
    cl.clReleaseMemObject(dev_w);
    cl.clReleaseMemObject(dev_sc);
    cl.clReleaseMemObject(dev_out);
    cl.clReleaseKernel(kernel);
}

void GpuEngine::flux_sgemm(int, int, int, const uint8_t*, const uint8_t*, float*) {}

} // namespace hlas
} // namespace mecan
