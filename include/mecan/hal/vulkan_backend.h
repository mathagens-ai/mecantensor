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

    typedef void* VkInstance;
    typedef void* VkPhysicalDevice;
    typedef int VkResult;

    #define VK_SUCCESS 0
    #define VK_STRUCTURE_TYPE_APPLICATION_INFO 0
    #define VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO 1
    #define VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU 1
    #define VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU 2

    struct VkApplicationInfo {
        int sType;
        const void* pNext;
        const char* pApplicationName;
        uint32_t applicationVersion;
        const char* pEngineName;
        uint32_t engineVersion;
        uint32_t apiVersion;
    };

    struct VkInstanceCreateInfo {
        int sType;
        const void* pNext;
        uint32_t flags;
        const VkApplicationInfo* pApplicationInfo;
        uint32_t enabledLayerCount;
        const char* const* ppEnabledLayerNames;
        uint32_t enabledExtensionCount;
        const char* const* ppEnabledExtensionNames;
    };

    struct VkPhysicalDeviceLimits {
        uint32_t maxImageDimension1D;
        uint32_t maxImageDimension2D;
        uint32_t maxImageDimension3D;
        uint32_t maxImageDimensionCube;
        uint32_t maxImageArrayLayers;
        uint32_t maxTexelBufferElements;
        uint32_t maxUniformBufferRange;
        uint32_t maxStorageBufferRange;
        uint32_t maxPushConstantsSize;
        uint32_t maxMemoryAllocationCount;
        uint32_t maxSamplerAllocationCount;
        uint64_t bufferImageGranularity;
        uint64_t sparseAddressSpaceSize;
        uint32_t maxBoundDescriptorSets;
        uint32_t maxPerStageDescriptorSamplers;
        uint32_t maxPerStageDescriptorUniformBuffers;
        uint32_t maxPerStageDescriptorStorageBuffers;
        uint32_t maxPerStageDescriptorSampledImages;
        uint32_t maxPerStageDescriptorStorageImages;
        uint32_t maxPerStageDescriptorInputAttachments;
        uint32_t maxPerStageResources;
        uint32_t maxDescriptorSetSamplers;
        uint32_t maxDescriptorSetUniformBuffers;
        uint32_t maxDescriptorSetUniformBuffersDynamic;
        uint32_t maxDescriptorSetStorageBuffers;
        uint32_t maxDescriptorSetStorageBuffersDynamic;
        uint32_t maxDescriptorSetSampledImages;
        uint32_t maxDescriptorSetStorageImages;
        uint32_t maxDescriptorSetInputAttachments;
        uint32_t maxVertexInputAttributes;
        uint32_t maxVertexInputBindings;
        uint32_t maxVertexInputAttributeOffset;
        uint32_t maxVertexInputBindingStride;
        uint32_t maxTypeStageOutputs;
        uint32_t maxComputeSharedMemorySize;
        uint32_t maxComputeWorkGroupCount[3];
        uint32_t maxComputeWorkGroupInvocations;
        uint32_t maxComputeWorkGroupSize[3];
    };

    struct VkPhysicalDeviceProperties {
        uint32_t apiVersion;
        uint32_t driverVersion;
        uint32_t vendorID;
        uint32_t deviceID;
        uint32_t deviceType;
        char deviceName[256];
        uint8_t pipelineCacheUUID[16];
        VkPhysicalDeviceLimits limits;
    };

    struct VkMemoryType {
        uint32_t propertyFlags;
        uint32_t heapIndex;
    };

    struct VkMemoryHeap {
        uint64_t size;
        uint32_t flags;
    };

    struct VkPhysicalDeviceMemoryProperties {
        uint32_t memoryTypeCount;
        VkMemoryType memoryTypes[32];
        uint32_t memoryHeapCount;
        VkMemoryHeap memoryHeaps[16];
    };

    class VulkanBackend : public Backend {
    public:
        VulkanBackend() : lib_(nullptr), available_(false) {
            probe();
        }

        ~VulkanBackend() override {
            // Library handle released at shutdown
        }

        std::string name() const override { return "MecanTensor Vulkan Compute Backend"; }
        BackendAPI  api() const override { return BackendAPI::VULKAN_COMPUTE; }
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
            lib_ = MECAN_LOAD_LIB("vulkan-1.dll");
#else
            lib_ = MECAN_LOAD_LIB("libvulkan.so.1");
            if (!lib_) lib_ = MECAN_LOAD_LIB("libvulkan.so");
#endif
            if (!lib_) {
                available_ = false;
                return;
            }

            typedef VkResult (*fn_vkCreateInstance)(const VkInstanceCreateInfo*, const void*, VkInstance*);
            typedef void (*fn_vkDestroyInstance)(VkInstance, const void*);
            typedef VkResult (*fn_vkEnumeratePhysicalDevices)(VkInstance, uint32_t*, VkPhysicalDevice*);
            typedef void (*fn_vkGetPhysicalDeviceProperties)(VkPhysicalDevice, VkPhysicalDeviceProperties*);
            typedef void (*fn_vkGetPhysicalDeviceMemoryProperties)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties*);

            auto vkCreateInstance = (fn_vkCreateInstance)MECAN_GET_PROC(lib_, "vkCreateInstance");
            auto vkDestroyInstance = (fn_vkDestroyInstance)MECAN_GET_PROC(lib_, "vkDestroyInstance");
            auto vkEnumeratePhysicalDevices = (fn_vkEnumeratePhysicalDevices)MECAN_GET_PROC(lib_, "vkEnumeratePhysicalDevices");
            auto vkGetPhysicalDeviceProperties = (fn_vkGetPhysicalDeviceProperties)MECAN_GET_PROC(lib_, "vkGetPhysicalDeviceProperties");
            auto vkGetPhysicalDeviceMemoryProperties = (fn_vkGetPhysicalDeviceMemoryProperties)MECAN_GET_PROC(lib_, "vkGetPhysicalDeviceMemoryProperties");

            if (!vkCreateInstance || !vkDestroyInstance || !vkEnumeratePhysicalDevices || !vkGetPhysicalDeviceProperties || !vkGetPhysicalDeviceMemoryProperties) {
                available_ = false;
                return;
            }

            VkApplicationInfo app_info = {};
            app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            app_info.pApplicationName = "MecanTensor Dynamic Discovery";
            app_info.applicationVersion = 1;
            app_info.pEngineName = "MecanTensor";
            app_info.engineVersion = 1;
            app_info.apiVersion = (1 << 22) | (2 << 12); // Vulkan 1.2

            VkInstanceCreateInfo create_info = {};
            create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            create_info.pApplicationInfo = &app_info;

            VkInstance instance = nullptr;
            if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
                available_ = false;
                return;
            }

            uint32_t device_count = 0;
            if (vkEnumeratePhysicalDevices(instance, &device_count, nullptr) != VK_SUCCESS || device_count == 0) {
                vkDestroyInstance(instance, nullptr);
                available_ = false;
                return;
            }

            std::vector<VkPhysicalDevice> devices(device_count);
            if (vkEnumeratePhysicalDevices(instance, &device_count, devices.data()) == VK_SUCCESS) {
                for (uint32_t i = 0; i < device_count; ++i) {
                    VkPhysicalDeviceProperties props;
                    vkGetPhysicalDeviceProperties(devices[i], &props);

                    HWDeviceInfo cap = {};
                    cap.name = props.deviceName;

                    // Vendor translation
                    if (props.vendorID == 0x10DE) cap.vendor = "NVIDIA";
                    else if (props.vendorID == 0x1002 || props.vendorID == 0x1022) cap.vendor = "AMD";
                    else if (props.vendorID == 0x8086) cap.vendor = "Intel";
                    else if (props.vendorID == 0x106B) cap.vendor = "Apple";
                    else cap.vendor = "Unknown";

                    cap.driver_version = "Vulkan Driver";
                    cap.device_class = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU || props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) 
                                       ? DeviceClass::GPU : DeviceClass::CUSTOM;
                    cap.backend = BackendAPI::VULKAN_COMPUTE;

                    VkPhysicalDeviceMemoryProperties mem_props;
                    vkGetPhysicalDeviceMemoryProperties(devices[i], &mem_props);
                    
                    uint64_t vram_bytes = 0;
                    for (uint32_t h = 0; h < mem_props.memoryHeapCount; ++h) {
                        // Keep track of the largest device-local heap size
                        if ((mem_props.memoryHeaps[h].flags & 1) && mem_props.memoryHeaps[h].size > vram_bytes) {
                            vram_bytes = mem_props.memoryHeaps[h].size;
                        }
                    }
                    cap.global_mem_bytes = vram_bytes;

                    cap.compute_units = props.limits.maxComputeWorkGroupCount[0];
                    cap.clock_mhz = 1500; // Estimated baseline clock
                    cap.warp_size = 32; // Default, varies between AMD (64) and NVIDIA (32)

                    cap.supports_fp64 = false;
                    cap.supports_fp16 = true;
                    cap.supports_bf16 = false;
                    cap.supports_int8 = true;
                    cap.supports_binary = true;

                    discovered_devices_.push_back(cap);
                }
            }

            vkDestroyInstance(instance, nullptr);
            available_ = !discovered_devices_.empty();
        }
    };

} // namespace hal
} // namespace mecan
