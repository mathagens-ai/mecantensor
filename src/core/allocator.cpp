#include "core/allocator.h"
#include <cstdlib>
#include <stdexcept>
#include <unordered_map>
#include <mutex>

#ifdef _WIN32
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif

namespace mecan {
namespace core {

    // 64-byte alignment for AVX-512 and cache line optimization
    constexpr size_t TST_ALIGNMENT = 64;

    void* CPUAllocator::allocate(size_t nbytes) {
        if (nbytes == 0) return nullptr;
        
        void* ptr = nullptr;
#ifdef _WIN32
        ptr = _aligned_malloc(nbytes, TST_ALIGNMENT);
#else
        if (posix_memalign(&ptr, TST_ALIGNMENT, nbytes) != 0) {
            ptr = nullptr;
        }
#endif
        
        if (!ptr) {
            throw std::runtime_error("TST Error: CPU memory allocation failed (OOM or Alignment error).");
        }
        return ptr;
    }

    void CPUAllocator::deallocate(void* ptr) {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {
    std::unordered_map<void*, size_t>& ssd_allocation_sizes() {
        static std::unordered_map<void*, size_t> table;
        return table;
    }
    std::mutex& ssd_allocation_mutex() {
        static std::mutex m;
        return m;
    }
}

    // SSD Infusion: Memory-mapped storage for massive parameter counts (50B+)
    void* SSDInfusionAllocator::allocate(size_t nbytes) {
#ifdef _WIN32
        // Windows File Mapping
        HANDLE hFile = CreateFileA("mt_infusion.bin", GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY, NULL);
        if (hFile == INVALID_HANDLE_VALUE) throw std::runtime_error("TST SSD Error: File creation failed.");
        
        HANDLE hMap = CreateFileMapping(hFile, NULL, PAGE_READWRITE, 0, (DWORD)nbytes, NULL);
        if (!hMap) { CloseHandle(hFile); throw std::runtime_error("TST SSD Error: Mapping failed."); }
        
        void* ptr = MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, nbytes);
        CloseHandle(hMap);
        CloseHandle(hFile);
        {
            std::lock_guard<std::mutex> lock(ssd_allocation_mutex());
            ssd_allocation_sizes()[ptr] = nbytes;
        }
        return ptr;
#else
        // Linux/Unix mmap
        int fd = open("/tmp/mt_infusion.bin", O_RDWR | O_CREAT | O_TRUNC, 0666);
        if (fd == -1) throw std::runtime_error("TST SSD Error: SSD file access failed.");
        ftruncate(fd, nbytes);
        void* ptr = mmap(NULL, nbytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (ptr == MAP_FAILED) throw std::runtime_error("TST SSD Error: mmap failed.");
        {
            std::lock_guard<std::mutex> lock(ssd_allocation_mutex());
            ssd_allocation_sizes()[ptr] = nbytes;
        }
        return ptr;
#endif
    }

    void SSDInfusionAllocator::deallocate(void* ptr) {
        if (!ptr) return;
        size_t nbytes = 0;
        {
            std::lock_guard<std::mutex> lock(ssd_allocation_mutex());
            auto it = ssd_allocation_sizes().find(ptr);
            if (it != ssd_allocation_sizes().end()) {
                nbytes = it->second;
                ssd_allocation_sizes().erase(it);
            }
        }
#ifdef _WIN32
        UnmapViewOfFile(ptr);
#else
        if (nbytes > 0) {
            munmap(ptr, nbytes);
        } else {
            // Fallback for unknown allocations.
            free(ptr);
        }
#endif
    }

    // Static global allocators
    static CPUAllocator g_cpu_alloc;
    static SSDInfusionAllocator g_ssd_alloc;

    Allocator* get_allocator(DeviceType device_type) {
        switch (device_type) {
            case DeviceType::CPU: return &g_cpu_alloc;
            case DeviceType::SSD_Infusion: return &g_ssd_alloc;
            default: return &g_cpu_alloc;
        }
    }

} // namespace core
} // namespace mecan
