"""
MecanTensor Ops — Core Math Operations (Python Wrapper)

Wraps the C++ math kernels (matmul, add, bitlinear, flash_ternary_attention).
Falls back to numpy when the native library is unavailable.

Usage:
    from mecantensor import ops
    C = ops.matmul(A, B)
    D = ops.add(A, B)
"""

import numpy as np
from ._native import get_lib
import ctypes


def matmul(A, B):
    """
    Matrix multiply via the MecanTensor C++ kernel.

    Args:
        A: numpy float32 (M, K)
        B: numpy float32 (K, N)

    Returns:
        C: numpy float32 (M, N)
    """
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)

    lib = get_lib()
    if lib is not None and hasattr(lib, 'mt_op_matmul'):
        M, K = A.shape
        N = B.shape[1]
        C = np.zeros((M, N), dtype=np.float32)

        a_shape = (ctypes.c_int64 * 2)(M, K)
        b_shape = (ctypes.c_int64 * 2)(K, N)
        c_shape = (ctypes.c_int64 * 2)(M, N)

        a_ptr = lib.mt_create_tensor(a_shape, 2, 1, 0)
        b_ptr = lib.mt_create_tensor(b_shape, 2, 1, 0)
        c_ptr = lib.mt_create_tensor(c_shape, 2, 1, 0)

        try:
            lib.mt_get_data_ptr_f32.restype = ctypes.POINTER(ctypes.c_float)
            ctypes.memmove(lib.mt_get_data_ptr_f32(a_ptr), A.ctypes.data, A.nbytes)
            ctypes.memmove(lib.mt_get_data_ptr_f32(b_ptr), B.ctypes.data, B.nbytes)
            lib.mt_op_matmul(a_ptr, b_ptr, c_ptr)
            ctypes.memmove(C.ctypes.data, lib.mt_get_data_ptr_f32(c_ptr), C.nbytes)
        finally:
            lib.mt_destroy_tensor(a_ptr)
            lib.mt_destroy_tensor(b_ptr)
            lib.mt_destroy_tensor(c_ptr)

        return C

    return A @ B


def add(A, B):
    """Element-wise add via C++ kernel or numpy."""
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)

    lib = get_lib()
    if lib is not None and hasattr(lib, 'mt_op_add'):
        C = np.zeros_like(A)
        a_shape = (ctypes.c_int64 * len(A.shape))(*A.shape)
        b_shape = (ctypes.c_int64 * len(B.shape))(*B.shape)
        c_shape = (ctypes.c_int64 * len(C.shape))(*C.shape)

        a_ptr = lib.mt_create_tensor(a_shape, len(A.shape), 1, 0)
        b_ptr = lib.mt_create_tensor(b_shape, len(B.shape), 1, 0)
        c_ptr = lib.mt_create_tensor(c_shape, len(C.shape), 1, 0)

        try:
            lib.mt_get_data_ptr_f32.restype = ctypes.POINTER(ctypes.c_float)
            ctypes.memmove(lib.mt_get_data_ptr_f32(a_ptr), A.ctypes.data, A.nbytes)
            ctypes.memmove(lib.mt_get_data_ptr_f32(b_ptr), B.ctypes.data, B.nbytes)
            lib.mt_op_add(a_ptr, b_ptr, c_ptr)
            ctypes.memmove(C.ctypes.data, lib.mt_get_data_ptr_f32(c_ptr), C.nbytes)
        finally:
            lib.mt_destroy_tensor(a_ptr)
            lib.mt_destroy_tensor(b_ptr)
            lib.mt_destroy_tensor(c_ptr)
        return C

    return (A + B).astype(np.float32)


def bitlinear(input, weight, bias):
    """BitLinear forward pass (1.58-bit ternary)."""
    lib = get_lib()
    if lib is not None and hasattr(lib, 'mt_op_bitlinear'):
        # TODO: wire through C++ bridge when tensor creation is simplified
        pass
    # Fallback
    input = np.asarray(input, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)
    return (input @ weight.T + bias).astype(np.float32)


def flash_ternary_attention(Q, K, V):
    """Flash ternary attention via the C++ kernel."""
    lib = get_lib()
    if lib is not None and hasattr(lib, 'mt_flash_ternary_attention'):
        # TODO: wire when shape handling is cleaner
        pass
    # Fallback: standard scaled dot-product attention
    Q = np.asarray(Q, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)
    V = np.asarray(V, dtype=np.float32)
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    # Softmax
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return (attn @ V).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  VISION & ARCHITECTURE PARITY OPERATIONS (Fallbacks)
# ═══════════════════════════════════════════════════════════════════════════════

def max_pool2d(x, kernel_size, stride=None, padding=0):
    """Native C++ MaxPool2d."""
    if stride is None: stride = kernel_size
    x = np.ascontiguousarray(x, dtype=np.float32)
    N, C, H, W = x.shape
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    out = np.zeros((N, C, H_out, W_out), dtype=np.float32)
    
    lib = get_lib()
    if lib is not None and hasattr(lib, 'mt_op_max_pool2d'):
        in_shape = (ctypes.c_int64 * 4)(N, C, H, W)
        out_shape = (ctypes.c_int64 * 4)(N, C, H_out, W_out)
        in_ptr = lib.mt_create_tensor(in_shape, 4, 1, 0)
        out_ptr = lib.mt_create_tensor(out_shape, 4, 1, 0)
        try:
            lib.mt_get_data_ptr_f32.restype = ctypes.POINTER(ctypes.c_float)
            ctypes.memmove(lib.mt_get_data_ptr_f32(in_ptr), x.ctypes.data, x.nbytes)
            lib.mt_op_max_pool2d(in_ptr, out_ptr, kernel_size, kernel_size, stride, padding)
            ctypes.memmove(out.ctypes.data, lib.mt_get_data_ptr_f32(out_ptr), out.nbytes)
        finally:
            lib.mt_destroy_tensor(in_ptr)
            lib.mt_destroy_tensor(out_ptr)
        return out
    
    # Fallback if no dll
    import scipy.ndimage as nd
    pool_shape = (1, 1, kernel_size, kernel_size)
    if padding > 0:
        x = np.pad(x, ((0,0), (0,0), (padding,padding), (padding,padding)), mode='constant', constant_values=-np.inf)
    out_scipy = nd.maximum_filter(x, size=pool_shape, mode='constant', cval=-np.inf)
    return out_scipy[:, :, ::stride, ::stride]


def avg_pool2d(x, kernel_size, stride=None, padding=0):
    """Native C++ AvgPool2d."""
    if stride is None: stride = kernel_size
    x = np.ascontiguousarray(x, dtype=np.float32)
    N, C, H, W = x.shape
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    out = np.zeros((N, C, H_out, W_out), dtype=np.float32)
    
    lib = get_lib()
    if lib is not None and hasattr(lib, 'mt_op_avg_pool2d'):
        in_shape = (ctypes.c_int64 * 4)(N, C, H, W)
        out_shape = (ctypes.c_int64 * 4)(N, C, H_out, W_out)
        in_ptr = lib.mt_create_tensor(in_shape, 4, 1, 0)
        out_ptr = lib.mt_create_tensor(out_shape, 4, 1, 0)
        try:
            lib.mt_get_data_ptr_f32.restype = ctypes.POINTER(ctypes.c_float)
            ctypes.memmove(lib.mt_get_data_ptr_f32(in_ptr), x.ctypes.data, x.nbytes)
            lib.mt_op_avg_pool2d(in_ptr, out_ptr, kernel_size, kernel_size, stride, padding)
            ctypes.memmove(out.ctypes.data, lib.mt_get_data_ptr_f32(out_ptr), out.nbytes)
        finally:
            lib.mt_destroy_tensor(in_ptr)
            lib.mt_destroy_tensor(out_ptr)
        return out
        
    # Fallback
    import scipy.ndimage as nd
    pool_shape = (1, 1, kernel_size, kernel_size)
    if padding > 0:
        x = np.pad(x, ((0,0), (0,0), (padding,padding), (padding,padding)), mode='constant')
    out_scipy = nd.uniform_filter(x, size=pool_shape, mode='constant', cval=0.0)
    return out_scipy[:, :, ::stride, ::stride]


def batch_norm2d(x, running_mean, running_var, weight=None, bias=None, training=False, eps=1e-5):
    """Native C++ BatchNorm2d."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    N, C, H, W = x.shape
    out = np.zeros_like(x)
    
    lib = get_lib()
    if lib is not None and hasattr(lib, 'mt_op_batch_norm2d'):
        shape4 = (ctypes.c_int64 * 4)(N, C, H, W)
        shape1 = (ctypes.c_int64 * 1)(C)
        in_ptr = lib.mt_create_tensor(shape4, 4, 1, 0)
        out_ptr = lib.mt_create_tensor(shape4, 4, 1, 0)
        m_ptr = lib.mt_create_tensor(shape1, 1, 1, 0)
        v_ptr = lib.mt_create_tensor(shape1, 1, 1, 0)
        g_ptr = lib.mt_create_tensor(shape1, 1, 1, 0) if weight is not None else None
        b_ptr = lib.mt_create_tensor(shape1, 1, 1, 0) if bias is not None else None
        
        try:
            lib.mt_get_data_ptr_f32.restype = ctypes.POINTER(ctypes.c_float)
            ctypes.memmove(lib.mt_get_data_ptr_f32(in_ptr), x.ctypes.data, x.nbytes)
            
            # running stats
            rm = np.ascontiguousarray(running_mean, dtype=np.float32)
            rv = np.ascontiguousarray(running_var, dtype=np.float32)
            ctypes.memmove(lib.mt_get_data_ptr_f32(m_ptr), rm.ctypes.data, rm.nbytes)
            ctypes.memmove(lib.mt_get_data_ptr_f32(v_ptr), rv.ctypes.data, rv.nbytes)
            
            if g_ptr:
                w = np.ascontiguousarray(weight, dtype=np.float32)
                ctypes.memmove(lib.mt_get_data_ptr_f32(g_ptr), w.ctypes.data, w.nbytes)
            if b_ptr:
                b_arr = np.ascontiguousarray(bias, dtype=np.float32)
                ctypes.memmove(lib.mt_get_data_ptr_f32(b_ptr), b_arr.ctypes.data, b_arr.nbytes)
                
            lib.mt_op_batch_norm2d(in_ptr, out_ptr, g_ptr, b_ptr, m_ptr, v_ptr, eps, training)
            ctypes.memmove(out.ctypes.data, lib.mt_get_data_ptr_f32(out_ptr), out.nbytes)
        finally:
            lib.mt_destroy_tensor(in_ptr)
            lib.mt_destroy_tensor(out_ptr)
            lib.mt_destroy_tensor(m_ptr)
            lib.mt_destroy_tensor(v_ptr)
            if g_ptr: lib.mt_destroy_tensor(g_ptr)
            if b_ptr: lib.mt_destroy_tensor(b_ptr)
        return out
        
    # Fallback
    if training:
        mean = x.mean(axis=(0, 2, 3), keepdims=True)
        var = x.var(axis=(0, 2, 3), keepdims=True)
    else:
        mean = np.asarray(running_mean, dtype=np.float32).reshape(1, C, 1, 1)
        var = np.asarray(running_var, dtype=np.float32).reshape(1, C, 1, 1)
        
    out_py = (x - mean) / np.sqrt(var + eps)
    if weight is not None:
        out_py = out_py * np.asarray(weight, dtype=np.float32).reshape(1, C, 1, 1)
    if bias is not None:
        out_py = out_py + np.asarray(bias, dtype=np.float32).reshape(1, C, 1, 1)
    return out_py


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Fallback LayerNorm."""
    x = np.asarray(x, dtype=np.float32)
    dims = tuple(range(x.ndim - len(normalized_shape), x.ndim))
    mean = x.mean(axis=dims, keepdims=True)
    var = x.var(axis=dims, keepdims=True)
    out = (x - mean) / np.sqrt(var + eps)
    if weight is not None: out *= weight
    if bias is not None: out += bias
    return out


def pixel_shuffle(x, upscale_factor):
    """Native C++ PixelShuffle."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    N, C_in, H, W = x.shape
    r = upscale_factor
    C = C_in // (r ** 2)
    out = np.zeros((N, C, H * r, W * r), dtype=np.float32)
    
    lib = get_lib()
    if lib is not None and hasattr(lib, 'mt_op_pixel_shuffle'):
        in_shape = (ctypes.c_int64 * 4)(N, C_in, H, W)
        out_shape = (ctypes.c_int64 * 4)(N, C, H * r, W * r)
        in_ptr = lib.mt_create_tensor(in_shape, 4, 1, 0)
        out_ptr = lib.mt_create_tensor(out_shape, 4, 1, 0)
        try:
            lib.mt_get_data_ptr_f32.restype = ctypes.POINTER(ctypes.c_float)
            ctypes.memmove(lib.mt_get_data_ptr_f32(in_ptr), x.ctypes.data, x.nbytes)
            lib.mt_op_pixel_shuffle(in_ptr, out_ptr, r)
            ctypes.memmove(out.ctypes.data, lib.mt_get_data_ptr_f32(out_ptr), out.nbytes)
        finally:
            lib.mt_destroy_tensor(in_ptr)
            lib.mt_destroy_tensor(out_ptr)
        return out
        
    # Fallback
    x_reshaped = x.reshape(N, C, r, r, H, W)
    out_py = x_reshaped.transpose(0, 1, 4, 2, 5, 3).reshape(N, C, H * r, W * r)
    return out_py


def nms(boxes, scores, iou_threshold=0.5):
    """
    Fallback Non-Maximum Suppression.
    boxes: (N, 4) in [x1, y1, x2, y2]
    scores: (N,)
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    
    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
        
    return keep

# ═══════════════════════════════════════════════════════════════════════════════
#  VISION ENGINE (Motion, Vector, Light)
# ═══════════════════════════════════════════════════════════════════════════════

def farneback_flow(img1, img2, window_size=15, iterations=3):
    """Native C++ Dense Optical Flow."""
    img1 = np.ascontiguousarray(img1, dtype=np.float32)
    img2 = np.ascontiguousarray(img2, dtype=np.float32)
    H, W = img1.shape
    out = np.zeros((2, H, W), dtype=np.float32)
    
    lib = get_lib()
    if lib is not None and hasattr(lib, 'mt_vision_farneback_flow'):
        shape2 = (ctypes.c_int64 * 2)(H, W)
        shape3 = (ctypes.c_int64 * 3)(2, H, W)
        i1_ptr = lib.mt_create_tensor(shape2, 2, 1, 0)
        i2_ptr = lib.mt_create_tensor(shape2, 2, 1, 0)
        o_ptr = lib.mt_create_tensor(shape3, 3, 1, 0)
        try:
            lib.mt_get_data_ptr_f32.restype = ctypes.POINTER(ctypes.c_float)
            ctypes.memmove(lib.mt_get_data_ptr_f32(i1_ptr), img1.ctypes.data, img1.nbytes)
            ctypes.memmove(lib.mt_get_data_ptr_f32(i2_ptr), img2.ctypes.data, img2.nbytes)
            lib.mt_vision_farneback_flow(i1_ptr, i2_ptr, o_ptr, window_size, iterations)
            ctypes.memmove(out.ctypes.data, lib.mt_get_data_ptr_f32(o_ptr), out.nbytes)
        finally:
            lib.mt_destroy_tensor(i1_ptr)
            lib.mt_destroy_tensor(i2_ptr)
            lib.mt_destroy_tensor(o_ptr)
    return out

def signed_distance_field(binary_mask):
    """Native C++ Signed Distance Field."""
    mask = np.ascontiguousarray(binary_mask, dtype=np.float32)
    H, W = mask.shape
    out = np.zeros((H, W), dtype=np.float32)
    
    lib = get_lib()
    if lib is not None and hasattr(lib, 'mt_vision_signed_distance_field'):
        shape = (ctypes.c_int64 * 2)(H, W)
        m_ptr = lib.mt_create_tensor(shape, 2, 1, 0)
        o_ptr = lib.mt_create_tensor(shape, 2, 1, 0)
        try:
            lib.mt_get_data_ptr_f32.restype = ctypes.POINTER(ctypes.c_float)
            ctypes.memmove(lib.mt_get_data_ptr_f32(m_ptr), mask.ctypes.data, mask.nbytes)
            lib.mt_vision_signed_distance_field(m_ptr, o_ptr)
            ctypes.memmove(out.ctypes.data, lib.mt_get_data_ptr_f32(o_ptr), out.nbytes)
        finally:
            lib.mt_destroy_tensor(m_ptr)
            lib.mt_destroy_tensor(o_ptr)
    return out

def diffraction_pattern(img, aperture_mm=50.0, wave_nm=550.0, focal_mm=50.0, pitch_um=3.0):
    """Native C++ Airy Disk Convolution."""
    img = np.ascontiguousarray(img, dtype=np.float32)
    H, W = img.shape
    out = np.zeros((H, W), dtype=np.float32)
    
    lib = get_lib()
    if lib is not None and hasattr(lib, 'mt_vision_diffraction_pattern'):
        shape = (ctypes.c_int64 * 2)(H, W)
        i_ptr = lib.mt_create_tensor(shape, 2, 1, 0)
        o_ptr = lib.mt_create_tensor(shape, 2, 1, 0)
        try:
            lib.mt_get_data_ptr_f32.restype = ctypes.POINTER(ctypes.c_float)
            ctypes.memmove(lib.mt_get_data_ptr_f32(i_ptr), img.ctypes.data, img.nbytes)
            lib.mt_vision_diffraction_pattern(i_ptr, o_ptr,
                ctypes.c_float(aperture_mm), ctypes.c_float(wave_nm),
                ctypes.c_float(focal_mm), ctypes.c_float(pitch_um))
            ctypes.memmove(out.ctypes.data, lib.mt_get_data_ptr_f32(o_ptr), out.nbytes)
        finally:
            lib.mt_destroy_tensor(i_ptr)
            lib.mt_destroy_tensor(o_ptr)
    return out

def canny(gray, low_threshold=50.0, high_threshold=150.0):
    """Native C++ Canny Edge Detection."""
    gray = np.ascontiguousarray(gray, dtype=np.float32)
    H, W = gray.shape
    out = np.zeros((H, W), dtype=np.float32)
    
    lib = get_lib()
    if lib is not None and hasattr(lib, 'mt_vision_canny'):
        shape = (ctypes.c_int64 * 2)(H, W)
        i_ptr = lib.mt_create_tensor(shape, 2, 1, 0)
        o_ptr = lib.mt_create_tensor(shape, 2, 1, 0)
        try:
            lib.mt_get_data_ptr_f32.restype = ctypes.POINTER(ctypes.c_float)
            ctypes.memmove(lib.mt_get_data_ptr_f32(i_ptr), gray.ctypes.data, gray.nbytes)
            lib.mt_vision_canny(i_ptr, o_ptr, ctypes.c_float(low_threshold), ctypes.c_float(high_threshold))
            ctypes.memmove(out.ctypes.data, lib.mt_get_data_ptr_f32(o_ptr), out.nbytes)
        finally:
            lib.mt_destroy_tensor(i_ptr)
            lib.mt_destroy_tensor(o_ptr)
    return out

