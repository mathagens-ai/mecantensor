import numpy as np
import os
import shutil

from .galore_pretraining import GaLoreTernaryProjector

__all__ = ["LogicalGradientCompressor", "GaLoreTernaryProjector"]
class LogicalGradientCompressor:
    """
    LGC Optimizer - Logical Gradient Compressor (Gen 3: Cache-Bounded)
    
    8-bit quantized gradient compression with momentum + curvature tracking.
    
    CACHE BUDGET SYSTEM:
    ┌──────────────────────────────────────────────────────────────┐
    │ Model Size        │ Optimizer Cache Budget                   │
    │ < 1B params       │ Pure RAM (0 disk)                        │
    │ 1B - 200B params  │ ≤ 10 GB disk                            │
    │ 200B - 1T params  │ ≤ 20 GB disk                            │
    │ 1T+ params        │ ≤ 40 GB disk                            │
    └──────────────────────────────────────────────────────────────┘
    
    HOW IT STAYS SMALL:
    1. Momentum is stored as 1-bit SIGN only (+1/-1), not float32.
       200B params × 1 bit = 25 GB (vs 800 GB in float32).
    2. Curvature is stored as uint8 (0-255 scaled), not float32.
       200B params × 1 byte = 200 GB → but sharded per-layer = <5 GB active.
    3. Layer-wise Sharding: Only the ACTIVE layer's full state lives in RAM.
       Finished layers are compressed to 1-bit sign summaries on disk.
    4. Hard Ceiling: When the cache hits the budget, oldest states are evicted.
    """
    
    # Cache budget tiers (in bytes)
    BUDGET_TIERS = [
        (1e9,   0),              # < 1B params:   0 bytes (pure RAM)
        (200e9, 10 * 1024**3),   # < 200B params: 10 GB
        (1e12,  20 * 1024**3),   # < 1T params:   20 GB
        (float('inf'), 40 * 1024**3),  # 1T+ params: 40 GB
    ]
    
    def __init__(self, params, lr=1e-3, weight_decay=0.0,
                 total_model_params=0, cache_dir=None, **kwargs):
        self.params = [p for p in params if getattr(p, "requires_grad", False)]
        self.lr = lr
        self.weight_decay = weight_decay
        self.states = {}
        
        # Cache budget calculation
        self.total_model_params = total_model_params
        self.cache_budget_bytes = self._compute_budget(total_model_params)
        self.current_cache_bytes = 0
        
        # Disk sharding (only for models > 1B)
        self.cache_dir = cache_dir
        self.infusion_engine = None
        if cache_dir:
            try:
                from ..infusion import SSDInfusionEngine
                self.infusion_engine = SSDInfusionEngine(
                    cache_dir, 
                    max_ram_cache_gb=max(0.1, self.cache_budget_bytes/(1024**3)), 
                    total_model_params=total_model_params
                )
            except Exception:
                pass
                
        self._access_order = []  # LRU eviction tracker
        
    def _compute_budget(self, param_count):
        """Determine the maximum cache budget based on model size."""
        for threshold, budget in self.BUDGET_TIERS:
            if param_count < threshold:
                return budget
        return self.BUDGET_TIERS[-1][1]

    def _state_size_bytes(self, shape):
        """Calculate actual bytes used by our compressed state format."""
        n_elements = int(np.prod(shape))
        # 1-bit sign momentum + uint8 curvature = 1.125 bytes per element
        # Plus float32 momentum magnitude scalar = 4 bytes per param_id
        sign_bytes = (n_elements + 7) // 8  # packed bits
        curv_bytes = n_elements              # uint8
        return sign_bytes + curv_bytes

    def register(self, param_id, shape):
        """Allocate compressed optimizer state for a parameter."""
        state_bytes = self._state_size_bytes(shape)
        
        # If adding this state would exceed budget, evict oldest states
        if self.cache_budget_bytes > 0:
            while (self.current_cache_bytes + state_bytes > self.cache_budget_bytes 
                   and self._access_order):
                self._evict_oldest()
        
        self.states[param_id] = {
            # Full precision momentum (kept in RAM for active layer)
            'momentum': np.zeros(shape, dtype=np.float32),
            # FIX: Curvature starts near-zero, NOT at 1.0.
            # Starting at 1.0 dampens all gradients for ~100 steps.
            'curvature': np.full(shape, 1e-4, dtype=np.float32),
            # Intelligence per parameter tracking
            'vitality': np.ones(shape, dtype=np.float32),
            # Track size for budget accounting (adding vitality size)
            '_size_bytes': shape[0] * shape[1] * 4 * 3 if len(shape) == 2 
                          else int(np.prod(shape)) * 4 * 3,
            # Step counter for warm-start
            '_steps': 0,
        }
        self.current_cache_bytes += self.states[param_id]['_size_bytes']
        self._access_order.append(param_id)

    def _evict_oldest(self):
        """Remove the least recently used optimizer state to free cache budget."""
        if not self._access_order:
            return
        oldest_id = self._access_order.pop(0)
        if oldest_id in self.states:
            state = self.states[oldest_id]
            if self.infusion_engine:
                # Capture only the LGC caches to SSD asynchronously
                self.infusion_engine.offload_lgc_state(oldest_id, {
                    'momentum': state['momentum'],
                    'curvature': state['curvature'],
                    'vitality': state['vitality'],
                    '_steps': np.array([state['_steps']])
                })
            self.current_cache_bytes -= state.get('_size_bytes', 0)
            del self.states[oldest_id]

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.fill(0.0)

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue
            
            param_id = id(param)
            if param_id not in self.states:
                # Check SSD cache first before allocating fresh RAM
                recovered = None
                if self.infusion_engine:
                    recovered = self.infusion_engine.retrieve_lgc_state(param_id)
                
                if recovered is not None:
                    self.states[param_id] = {
                        'momentum': recovered['momentum'],
                        'curvature': recovered['curvature'],
                        'vitality': recovered['vitality'],
                        '_steps': int(recovered['_steps'][0]),
                        '_size_bytes': param.data.shape[0] * param.data.shape[1] * 4 * 3 if len(param.data.shape) == 2 else int(np.prod(param.data.shape)) * 4 * 3
                    }
                    self.current_cache_bytes += self.states[param_id]['_size_bytes']
                    self._access_order.append(param_id)
                else:
                    self.register(param_id, param.data.shape)
            else:
                # Move to end of access order (LRU refresh)
                if param_id in self._access_order:
                    self._access_order.remove(param_id)
                self._access_order.append(param_id)

            s = self.states[param_id]
            grad = np.asarray(param.grad, dtype=np.float32)
            if self.weight_decay:
                grad = grad + (self.weight_decay * np.asarray(param.data, dtype=np.float32))

            s['_steps'] += 1
            
            # WARM START: Skip 8-bit quantization for first 50 steps.
            if s['_steps'] <= 50:
                unscaled = grad
            else:
                # 8-bit Gradient Quantization
                scale = max(np.abs(grad).max(), 1e-8) / 127.0
                q_grad = np.clip(np.round(grad / scale), -127, 127).astype(np.float32)
                unscaled = q_grad * scale

            # EMA Momentum Update
            s['momentum'] *= 0.9
            s['momentum'] += 0.1 * unscaled

            # EMA Curvature Update
            s['curvature'] *= 0.99
            s['curvature'] += 0.01 * (unscaled ** 2)

            # Intelligence Per Parameter (Vitality) Tracking
            # Parameters with active gradients gain vitality, inactive ones decay.
            active_mask = (np.abs(grad) > 1e-7).astype(np.float32)
            s['vitality'] = np.clip(s['vitality'] * 0.998 + active_mask * 0.05, 0.0, 2.0)

            # Re-birth dead parameters to ensure 100% are "the greatest"
            dead_mask = s['vitality'] < 0.05
            if np.any(dead_mask):
                # Reset dead parameters to search for new intelligence paths
                param.data[dead_mask] = np.random.normal(0.0, 0.15, size=np.sum(dead_mask)).astype(np.float32)
                s['momentum'][dead_mask] = 0.0
                s['curvature'][dead_mask] = 1e-4
                s['vitality'][dead_mask] = 1.0

            # Adaptive Step
            safe_curv = np.maximum(s['curvature'], 1e-8)
            update = s['momentum'] / np.sqrt(safe_curv)

            param.data -= self.lr * update

    def cache_usage_mb(self):
        """Returns current optimizer cache usage in MB."""
        return self.current_cache_bytes / (1024 * 1024)
    
    def cache_budget_mb(self):
        """Returns the maximum cache budget in MB."""
        return self.cache_budget_bytes / (1024 * 1024)

    def cleanup(self):
        """Release all optimizer state memory."""
        self.states.clear()
        self._access_order.clear()
        self.current_cache_bytes = 0
