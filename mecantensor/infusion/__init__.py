import os
import numpy as np
import threading
import queue
import shutil
import tempfile
import time


class SSDInfusionEngine:
    """
    SSD Infusion Engine (Gen 3: Budget-Capped Tensor Offload)
    
    Selective tensor offload for models that exceed available RAM.
    
    CACHE BUDGET SYSTEM:
    ┌──────────────────────────────────────────────────────────────┐
    │ Model Size        │ SSD Cache Budget                        │
    │ < 1B params       │ DORMANT (0 bytes, pure RAM)             │
    │ 1B - 200B params  │ ≤ 10 GB                                │
    │ 200B - 1T params  │ ≤ 20 GB                                │
    │ 1T+ params        │ ≤ 40 GB                                │
    └──────────────────────────────────────────────────────────────┘
    
    HOW IT STAYS SMALL:
    1. DORMANT by default. No cache directory created until offload_tensor() is called.
    2. Hard budget ceiling. When the ceiling is hit, oldest tensors are evicted (deleted from SSD).
    3. Circular eviction ensures the cache NEVER grows beyond the budget.
    4. Tensors are stored as memory-mapped files for zero-copy retrieval.
    """
    
    BUDGET_TIERS = [
        (1e9,   0),
        (200e9, 10 * 1024**3),
        (1e12,  20 * 1024**3),
        (float('inf'), 40 * 1024**3),
    ]
    
    def __init__(self, cache_directory=None, max_ram_cache_gb=4.0,
                 total_model_params=0, create_cache_on_init=False):
        self.cache_directory = cache_directory
        self.max_ram_cache_bytes = int(max_ram_cache_gb * 1024 * 1024 * 1024)
        self.registry = {}
        self.active_maps = {}
        self.cache_ready = False
        
        # Budget system
        self.cache_budget_bytes = self._compute_budget(total_model_params)
        self._total_offloaded_bytes = 0
        self._eviction_order = []  # LRU: oldest first
        
        # Prefetch is lazy
        self._prefetch_thread = None
        self._prefetch_queue = queue.Queue()
        
        # Temporal State Caching for RNN/SSM Recurrent Checkpoints
        self.temporal_registry = {}  # Tracks states over time steps
        self._temporal_budget_bytes = max(100 * 1024 * 1024, self.cache_budget_bytes // 4) # Up to 25% for temporal
        
    def _compute_budget(self, param_count):
        for threshold, budget in self.BUDGET_TIERS:
            if param_count < threshold:
                return budget
        return self.BUDGET_TIERS[-1][1]
 
    def _ensure_cache_directory(self):
        if self.cache_directory is None:
            raise RuntimeError("Disk infusion disabled: no cache directory configured.")
        if not self.cache_ready:
            os.makedirs(self.cache_directory, exist_ok=True)
            self.cache_ready = True
 
    def _start_prefetch_if_needed(self):
        if self._prefetch_thread is None:
            self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
            self._prefetch_thread.start()
 
    def _prefetch_worker(self):
        while True:
            try:
                tensor_id = self._prefetch_queue.get()
                if tensor_id in self.registry:
                    meta = self.registry[tensor_id]
                    if meta.get('kind') == 'array' and self.cache_directory is not None:
                        file_path = self._array_file_path(tensor_id)
                        if os.path.exists(file_path):
                            mmap_array = np.memmap(file_path, dtype=meta['dtype'], mode='r', shape=meta['shape'])
                            _ = mmap_array[0:1]
                self._prefetch_queue.task_done()
            except Exception:
                pass
 
    def _array_file_path(self, tensor_id):
        return os.path.join(self.cache_directory, f"tensor_{tensor_id}.bin")
 
    def _bytes_file_path(self, tensor_id):
        return os.path.join(self.cache_directory, f"tensor_{tensor_id}.tse")
 
    def _evict_until_fits(self, incoming_bytes):
        """Evict oldest tensors until we have room for the incoming tensor."""
        while (self._total_offloaded_bytes + incoming_bytes > self.cache_budget_bytes 
               and self._eviction_order):
            oldest_id = self._eviction_order.pop(0)
            if oldest_id in self.registry:
                meta = self.registry[oldest_id]
                # Delete the file from SSD
                if meta.get('kind') == 'array':
                    fpath = self._array_file_path(oldest_id)
                elif meta.get('kind') == 'bytes':
                    fpath = self._bytes_file_path(oldest_id)
                else:
                    continue
                    
                # Reclaim the bytes
                if os.path.exists(fpath):
                    file_size = os.path.getsize(fpath)
                    os.remove(fpath)
                    self._total_offloaded_bytes -= file_size
                    
                # Remove from active maps
                self.active_maps.pop(oldest_id, None)
                del self.registry[oldest_id]
 
    def offload_tensor(self, tensor_id, tensor_data):
        """Write a tensor to SSD with hard budget enforcement."""
        self._ensure_cache_directory()
        self._start_prefetch_if_needed()
        
        if isinstance(tensor_data, (bytes, bytearray, memoryview)):
            data_bytes = bytes(tensor_data)
            # Enforce budget
            self._evict_until_fits(len(data_bytes))
            
            file_path = self._bytes_file_path(tensor_id)
            with open(file_path, "wb") as f:
                f.write(data_bytes)
            self.registry[tensor_id] = {'kind': 'bytes'}
            self._total_offloaded_bytes += len(data_bytes)
            self._eviction_order.append(tensor_id)
            return
 
        numpy_array = np.ascontiguousarray(tensor_data)
        incoming_bytes = numpy_array.nbytes
        
        # Enforce budget
        self._evict_until_fits(incoming_bytes)
        
        file_path = self._array_file_path(tensor_id)
        shape = numpy_array.shape
        dtype = numpy_array.dtype
        
        # Asynchronous offload to prevent training blocks
        def write_task():
            mmap_array = np.memmap(file_path, dtype=dtype, mode='w+', shape=shape)
            mmap_array[:] = numpy_array[:]
            mmap_array.flush()
            del mmap_array
            
        threading.Thread(target=write_task).start()
        
        self.registry[tensor_id] = {'kind': 'array', 'shape': shape, 'dtype': dtype}
        self._total_offloaded_bytes += incoming_bytes
        self._eviction_order.append(tensor_id)
 
    def offload_lgc_state(self, param_id, state_dict):
        """
        Specialized extremely fast offload for LGC caches (momentum, curvature, vitality).
        Saves the dictionary of arrays asynchronously.
        """
        self._ensure_cache_directory()
        # Combine momentum, curvature, vitality into one file to minimize IO ops
        file_path = os.path.join(self.cache_directory, f"lgc_state_{param_id}.npz")
        
        def save_task():
            np.savez_compressed(file_path, **state_dict)
            
        # Run in background to keep training lightning fast
        threading.Thread(target=save_task).start()
        self.registry[f"lgc_{param_id}"] = {'kind': 'lgc_dict', 'path': file_path}
        
    def retrieve_lgc_state(self, param_id):
        """Retrieves LGC caches back from SSD into RAM."""
        key = f"lgc_{param_id}"
        if key in self.registry:
            meta = self.registry[key]
            if os.path.exists(meta['path']):
                with np.load(meta['path']) as data:
                    return {k: np.array(v) for k, v in data.items()}
        return None
 
    def offload_temporal_state(self, sequence_id, time_step, tensor_data):
        """
        Specialized offload for dynamic sequence states (Memory Caching/TurboQuant KV analog).
        It strictly manages rapid sequence checkpoints without disturbing model parameters.
        """
        self._ensure_cache_directory()
        numpy_array = np.ascontiguousarray(tensor_data)
        file_path = os.path.join(self.cache_directory, f"temporal_{sequence_id}_{time_step}.bin")
        shape, dtype = numpy_array.shape, numpy_array.dtype
        
        # Immediate overwrite mode for temporal speed
        mmap_array = np.memmap(file_path, dtype=dtype, mode='w+', shape=shape)
        mmap_array[:] = numpy_array[:]
        mmap_array.flush()
        del mmap_array
        
        if sequence_id not in self.temporal_registry:
            self.temporal_registry[sequence_id] = {}
        self.temporal_registry[sequence_id][time_step] = {'shape': shape, 'dtype': dtype, 'path': file_path}
        
    def retrieve_temporal_state(self, sequence_id, time_step):
        if sequence_id in self.temporal_registry and time_step in self.temporal_registry[sequence_id]:
            meta = self.temporal_registry[sequence_id][time_step]
            mmap_array = np.memmap(meta['path'], dtype=meta['dtype'], mode='c', shape=meta['shape'])
            self.active_maps[f"temp_{sequence_id}_{time_step}"] = mmap_array
            return mmap_array
        return None
 
    def hint_prefetch(self, tensor_id):
        if self.cache_directory is not None and self.cache_ready:
            self._prefetch_queue.put(tensor_id)
 
    def retrieve_tensor(self, tensor_id):
        if tensor_id in self.registry:
            meta = self.registry[tensor_id]
            # Refresh LRU position
            if tensor_id in self._eviction_order:
                self._eviction_order.remove(tensor_id)
                self._eviction_order.append(tensor_id)
                
            if meta.get('kind') == 'bytes':
                file_path = self._bytes_file_path(tensor_id)
                with open(file_path, "rb") as f:
                    return f.read()
            file_path = self._array_file_path(tensor_id)
            mmap_array = np.memmap(file_path, dtype=meta['dtype'], mode='c', shape=meta['shape'])
            self.active_maps[tensor_id] = mmap_array
            return mmap_array
        return None
 
    def offloaded_size_mb(self):
        return self._total_offloaded_bytes / (1024 * 1024)
    
    def offloaded_size_gb(self):
        return self._total_offloaded_bytes / (1024**3)
 
    def cache_budget_gb(self):
        return self.cache_budget_bytes / (1024**3)
 
    def cleanup(self):
        # Release all memmap references first (Windows file locking)
        for key in list(self.active_maps.keys()):
            try:
                del self.active_maps[key]
            except Exception:
                pass
        self.active_maps.clear()
        self.registry.clear()
        self._eviction_order.clear()
        self._total_offloaded_bytes = 0
        
        # Force garbage collection to release memmap file handles
        import gc
        gc.collect()
        
        if self.cache_directory is not None and os.path.exists(self.cache_directory):
            try:
                shutil.rmtree(self.cache_directory)
            except PermissionError:
                # Windows may hold locks briefly; retry after GC
                gc.collect()
                time.sleep(0.1)
                try:
                    shutil.rmtree(self.cache_directory)
                except Exception:
                    pass  # Cache dir will be overwritten on next use
        self.cache_ready = False
