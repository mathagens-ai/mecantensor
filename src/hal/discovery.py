"""
================================================================================
 MecanTensor Universal GPU/NPU/APU Discovery Engine
================================================================================
 Detects ANY compute device on the planet:
   - OpenCL: Intel, AMD, NVIDIA, ARM Mali, Qualcomm Adreno, Xilinx FPGAs
   - CUDA: NVIDIA GPUs (if driver present)
   - Vulkan: Nearly universal (phones, desktops, consoles)
   - DirectML: Windows ML accelerators (NPUs, GPUs)
   - CPU: Always available as fallback

 After detection, provides three execution profiles:
   Profile 1: CPU-only     (safest, always works)
   Profile 2: GPU-only     (maximum parallel throughput)
   Profile 3: CPU+GPU      (hybrid, splits work for max utilization)
================================================================================
"""
import os
import sys
import ctypes
import platform
import subprocess

class ComputeDevice:
    """Represents a single compute device."""
    def __init__(self):
        self.name = "Unknown"
        self.vendor = "Unknown"
        self.type = "Unknown"         # CPU, GPU, NPU, FPGA, APU
        self.backend = "None"         # OpenCL, CUDA, Vulkan, DirectML
        self.compute_units = 0
        self.clock_mhz = 0
        self.vram_bytes = 0
        self.local_mem_bytes = 0
        self.max_workgroup = 0
        self.supports_fp32 = True
        self.supports_fp16 = False
        self.supports_int8 = False
        self.supports_popcount = False  # Critical for QSBits
        self.driver_version = ""
        self.platform_name = ""
        self.cl_device = None         # OpenCL device object (if OpenCL)
        self.cl_context = None        # OpenCL context (if OpenCL)
        self.cl_queue = None          # OpenCL command queue (if OpenCL)

    def __repr__(self):
        vram = f"{self.vram_bytes / 1024**3:.1f} GB" if self.vram_bytes > 0 else "shared"
        return f"[{self.backend}] {self.name} ({self.vendor}) | {self.type} | {self.compute_units} CU @ {self.clock_mhz} MHz | VRAM: {vram}"


class UniversalDiscovery:
    """Discovers all compute devices available on this machine."""
    
    def __init__(self):
        self.devices = []
        self.cpu_devices = []
        self.gpu_devices = []
        self.npu_devices = []
        self.has_opencl = False
        self.has_cuda = False
        self.has_vulkan = False
        
    def discover_all(self):
        """Run all detection backends."""
        print("="*90)
        print(" MECANTENSOR UNIVERSAL COMPUTE DISCOVERY")
        print("="*90)
        
        self._detect_cpu()
        self._detect_opencl()
        self._detect_cuda()
        self._detect_vulkan()
        self._detect_directml()
        
        # Categorize
        self.cpu_devices = [d for d in self.devices if d.type == "CPU"]
        self.gpu_devices = [d for d in self.devices if d.type == "GPU"]
        self.npu_devices = [d for d in self.devices if d.type in ("NPU", "FPGA")]
        
        self._print_summary()
        return self.devices
    
    def _detect_cpu(self):
        """Detect CPU — always available."""
        d = ComputeDevice()
        d.type = "CPU"
        d.backend = "Native"
        d.supports_fp32 = True
        d.supports_int8 = True
        d.supports_popcount = True
        
        if platform.system() == "Windows":
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                d.name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
                d.clock_mhz = winreg.QueryValueEx(key, "~MHz")[0]
                winreg.CloseKey(key)
            except:
                d.name = platform.processor() or "Unknown CPU"
                d.clock_mhz = 0
        else:
            d.name = platform.processor() or "Unknown CPU"
        
        d.compute_units = os.cpu_count() or 1
        d.vendor = "Intel" if "intel" in d.name.lower() else (
                   "AMD" if "amd" in d.name.lower() else (
                   "Apple" if "apple" in d.name.lower() else (
                   "ARM" if "arm" in d.name.lower() else "Unknown")))
        
        # Detect SIMD capabilities
        d.supports_fp16 = False  # CPU fp16 is limited
        d.driver_version = platform.platform()
        
        self.devices.append(d)
        print(f"\n  [CPU] {d.name}")
        print(f"        {d.compute_units} cores @ {d.clock_mhz} MHz | Vendor: {d.vendor}")
    
    def _detect_opencl(self):
        """Detect OpenCL devices — covers Intel/AMD/NVIDIA/ARM/Qualcomm GPUs + FPGAs + NPUs."""
        try:
            import pyopencl as cl
            self.has_opencl = True
        except ImportError:
            print("\n  [OpenCL] Not available (pyopencl not installed)")
            return
        
        try:
            platforms = cl.get_platforms()
        except:
            print("\n  [OpenCL] No platforms found")
            return
        
        print(f"\n  [OpenCL] Found {len(platforms)} platform(s)")
        
        for plat in platforms:
            try:
                devices = plat.get_devices()
            except:
                continue
            
            for dev in devices:
                d = ComputeDevice()
                d.name = dev.name.strip()
                d.vendor = dev.vendor.strip()
                d.backend = "OpenCL"
                d.platform_name = plat.name.strip()
                d.compute_units = dev.max_compute_units
                d.clock_mhz = dev.max_clock_frequency
                d.vram_bytes = dev.global_mem_size
                d.local_mem_bytes = dev.local_mem_size
                d.max_workgroup = dev.max_work_group_size
                d.driver_version = dev.driver_version.strip() if hasattr(dev, 'driver_version') else ""
                
                # Determine device type
                dev_type = cl.device_type.to_string(dev.type)
                if "GPU" in dev_type:
                    d.type = "GPU"
                elif "CPU" in dev_type:
                    d.type = "CPU"  # OpenCL can also expose CPUs
                elif "ACCELERATOR" in dev_type or "CUSTOM" in dev_type:
                    d.type = "NPU"
                else:
                    d.type = "GPU"
                
                # Check capabilities
                extensions = dev.extensions.lower() if hasattr(dev, 'extensions') else ""
                d.supports_fp16 = "cl_khr_fp16" in extensions
                d.supports_int8 = True  # Most GPUs support int8
                d.supports_popcount = True  # OpenCL has popcount() built-in
                d.supports_fp32 = True
                
                # Store OpenCL objects for compute
                try:
                    d.cl_device = dev
                    d.cl_context = cl.Context([dev])
                    d.cl_queue = cl.CommandQueue(d.cl_context,
                        properties=cl.command_queue_properties.PROFILING_ENABLE)
                except:
                    pass
                
                self.devices.append(d)
                vram = f"{d.vram_bytes/1024**3:.1f} GB"
                print(f"    [{d.type}] {d.name} ({d.vendor})")
                print(f"         {d.compute_units} CU @ {d.clock_mhz} MHz | VRAM: {vram} | Local: {d.local_mem_bytes//1024} KB")
    
    def _detect_cuda(self):
        """Detect NVIDIA CUDA GPUs."""
        cuda_dll = None
        search = ["nvcuda.dll", "libcuda.so", "libcuda.dylib"]
        
        for name in search:
            try:
                cuda_dll = ctypes.CDLL(name)
                break
            except:
                continue
        
        if cuda_dll is None:
            # Try nvidia-smi
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                        "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    self.has_cuda = True
                    for line in result.stdout.strip().split('\n'):
                        parts = [x.strip() for x in line.split(',')]
                        if len(parts) >= 3:
                            d = ComputeDevice()
                            d.name = parts[0]
                            d.vendor = "NVIDIA"
                            d.type = "GPU"
                            d.backend = "CUDA"
                            d.vram_bytes = int(float(parts[1])) * 1024 * 1024
                            d.driver_version = parts[2]
                            d.supports_fp16 = True
                            d.supports_int8 = True
                            d.supports_popcount = True
                            self.devices.append(d)
                            print(f"\n  [CUDA] {d.name} | VRAM: {d.vram_bytes/1024**3:.1f} GB | Driver: {d.driver_version}")
                    return
            except:
                pass
            
            print("\n  [CUDA] Not available (no NVIDIA GPU detected)")
            return
        
        self.has_cuda = True
        print("\n  [CUDA] NVIDIA driver found")
    
    def _detect_vulkan(self):
        """Detect Vulkan compute devices."""
        vulkan_dll = None
        search = ["vulkan-1.dll", "libvulkan.so.1", "libvulkan.dylib",
                   "libMoltenVK.dylib"]
        
        for name in search:
            try:
                vulkan_dll = ctypes.CDLL(name)
                self.has_vulkan = True
                print(f"\n  [Vulkan] Runtime found ({name})")
                return
            except:
                continue
        
        print("\n  [Vulkan] Not available")
    
    def _detect_directml(self):
        """Detect Windows DirectML devices (NPUs, GPUs)."""
        if platform.system() != "Windows":
            return
        
        try:
            dml = ctypes.CDLL("DirectML.dll")
            print("\n  [DirectML] Runtime found (Windows ML accelerators)")
        except:
            # Check for NPU via Windows device manager
            try:
                result = subprocess.run(
                    ["powershell", "-Command",
                     "Get-PnpDevice -Class 'System' | Where-Object {$_.FriendlyName -match 'NPU|Neural'} | Select-Object -ExpandProperty FriendlyName"],
                    capture_output=True, text=True, timeout=5)
                if result.stdout.strip():
                    d = ComputeDevice()
                    d.name = result.stdout.strip().split('\n')[0]
                    d.vendor = "Intel" if "intel" in d.name.lower() else "Unknown"
                    d.type = "NPU"
                    d.backend = "DirectML"
                    self.devices.append(d)
                    print(f"\n  [DirectML/NPU] {d.name}")
                else:
                    print("\n  [DirectML] Not available")
            except:
                print("\n  [DirectML] Not available")
    
    def _print_summary(self):
        """Print discovery summary."""
        print(f"\n{'='*90}")
        print(f" DISCOVERY COMPLETE: {len(self.devices)} device(s) found")
        print(f"{'='*90}")
        print(f"  CPUs: {len(self.cpu_devices)}  |  GPUs: {len(self.gpu_devices)}  |  NPUs: {len(self.npu_devices)}")
        print(f"  Backends: OpenCL={'YES' if self.has_opencl else 'NO'}  CUDA={'YES' if self.has_cuda else 'NO'}  Vulkan={'YES' if self.has_vulkan else 'NO'}")
    
    def get_best_gpu(self):
        """Return the GPU with the most compute units."""
        if not self.gpu_devices:
            return None
        return max(self.gpu_devices, key=lambda d: d.compute_units)
    
    def get_cpu(self):
        """Return the primary CPU."""
        return self.cpu_devices[0] if self.cpu_devices else None


class ExecutionProfile:
    """Manages CPU-only, GPU-only, and CPU+GPU hybrid execution."""
    
    CPU_ONLY = "CPU"
    GPU_ONLY = "GPU"
    HYBRID = "CPU+GPU"
    
    def __init__(self, discovery: UniversalDiscovery):
        self.discovery = discovery
        self.cpu = discovery.get_cpu()
        self.gpu = discovery.get_best_gpu()
        self.available_profiles = self._detect_profiles()
    
    def _detect_profiles(self):
        profiles = []
        if self.cpu:
            profiles.append(self.CPU_ONLY)
        if self.gpu and self.gpu.cl_context:
            profiles.append(self.GPU_ONLY)
        if self.cpu and self.gpu and self.gpu.cl_context:
            profiles.append(self.HYBRID)
        return profiles
    
    def print_profiles(self):
        print(f"\n{'='*90}")
        print(f" EXECUTION PROFILES")
        print(f"{'='*90}")
        for p in self.available_profiles:
            if p == self.CPU_ONLY:
                print(f"  [x] CPU-ONLY   : {self.cpu.name} ({self.cpu.compute_units} cores)")
                print(f"                   Best for: small models, low-latency, guaranteed compatibility")
            elif p == self.GPU_ONLY:
                vram = f"{self.gpu.vram_bytes/1024**3:.1f} GB"
                print(f"  [x] GPU-ONLY   : {self.gpu.name} ({self.gpu.compute_units} CU, {vram})")
                print(f"                   Best for: large models, high throughput, batch inference")
            elif p == self.HYBRID:
                print(f"  [x] CPU+GPU    : {self.cpu.name} + {self.gpu.name}")
                print(f"                   Best for: maximum utilization, pipeline overlap")
        
        missing = [self.CPU_ONLY, self.GPU_ONLY, self.HYBRID]
        for p in self.available_profiles:
            missing.remove(p)
        for p in missing:
            print(f"  [ ] {p:<10} : Not available")
        
        print(f"{'='*90}")


# ── Main Entry Point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    disco = UniversalDiscovery()
    disco.discover_all()
    
    profiles = ExecutionProfile(disco)
    profiles.print_profiles()
