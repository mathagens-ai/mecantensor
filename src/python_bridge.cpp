#include "python_bridge.h"
#include "mecan/mecan.h"
#include "mecan/io/serialization.h"
#include <vector>
#include <utility>

#include "mecan/ops/pad.h"
#include "mecan/ops/conv.h"
#include "mecan/ops/pool.h"
#include "mecan/ops/norm.h"
#include "mecan/ops/upsample.h"
#include "mecan/vision/motion.h"
#include "mecan/vision/vector.h"
#include "mecan/vision/light.h"
#include "mecan/vision/detect.h"
#include "mecan/vision/color.h"
#include "mecan/vision/core_vision.h"

// Forward declarations for object_engine.cpp
namespace mecan { namespace vision { namespace object {
    void hog_descriptor(const float*, size_t, size_t, float*, int, int);
    void lbp(const float*, float*, size_t, size_t);
    float edge_density(const float*, size_t, size_t);
    int corner_count(const float*, size_t, size_t, float);
    float shape_complexity(const float*, size_t, size_t);
    float texture_energy(const float*, size_t, size_t);
    void bilateral_filter(const float*, float*, size_t, size_t, int, float, float);
    void morph_erode(const float*, float*, size_t, size_t, int);
    void morph_dilate(const float*, float*, size_t, size_t, int);
    void adaptive_threshold(const float*, float*, size_t, size_t, int, float);
    void gradient_map(const float*, float*, float*, size_t, size_t);
}}}

// Forward declarations for temporal engine (motion_engine.cpp)
namespace mecan { namespace vision { namespace temporal {
    float scene_change_score(const float*, const float*, size_t, size_t);
    void motion_heatmap(const float*, const float*, float*, size_t, size_t, float);
    float frame_stability(const float*, const float*, size_t, size_t);
    void flow_statistics(const float*, size_t, size_t, float*, float*, float*);
}}}

#if defined(_WIN32)
#define MECAN_EXPORT __declspec(dllexport)
#else
#define MECAN_EXPORT
#endif

extern "C" {

MECAN_EXPORT void* mt_create_tensor(int64_t* shape_ptr, int ndim, int dtype, int device) {
        std::vector<size_t> shape;
        for (int i = 0; i < ndim; ++i) shape.push_back((size_t)shape_ptr[i]);
        
        mecan::Tensor* T = new mecan::Tensor(shape, (mecan::core::ScalarType)dtype, mecan::core::Device((mecan::core::DeviceType)device));
        return (void*)T;
    }

MECAN_EXPORT void mt_destroy_tensor(void* tensor_ptr) {
        mecan::Tensor* T = static_cast<mecan::Tensor*>(tensor_ptr);
        delete T;
    }

MECAN_EXPORT float* mt_get_data_ptr_f32(void* tensor_ptr) {
        mecan::Tensor* T = static_cast<mecan::Tensor*>(tensor_ptr);
        return T->data_ptr<float>();
    }

MECAN_EXPORT uint8_t* mt_get_data_ptr_u8(void* tensor_ptr) {
        mecan::Tensor* T = static_cast<mecan::Tensor*>(tensor_ptr);
        return T->data_ptr<uint8_t>();
    }

MECAN_EXPORT void mt_op_add(void* a, void* b, void* out) {
        mecan::Tensor* A = static_cast<mecan::Tensor*>(a);
        mecan::Tensor* B = static_cast<mecan::Tensor*>(b);
        mecan::Tensor* OUT = static_cast<mecan::Tensor*>(out);
        mecan::ops::add(*A, *B, *OUT);
    }

MECAN_EXPORT void mt_op_matmul(void* a, void* b, void* out) {
        mecan::Tensor* A = static_cast<mecan::Tensor*>(a);
        mecan::Tensor* B = static_cast<mecan::Tensor*>(b);
        mecan::Tensor* OUT = static_cast<mecan::Tensor*>(out);
        mecan::ops::matmul(*A, *B, *OUT);
    }

MECAN_EXPORT int64_t mt_get_numel(void* tensor_ptr) {
        mecan::Tensor* T = static_cast<mecan::Tensor*>(tensor_ptr);
        return (int64_t)T->numel();
    }

MECAN_EXPORT void mt_save_paged_mt(
        void* tensor_ptr,
        const char* filepath,
        const char* tensor_name,
        int quant_scheme,
        uint64_t page_bytes) {
        mecan::Tensor* T = static_cast<mecan::Tensor*>(tensor_ptr);
        mecan::runtime::PagedTensorMetadata meta{};
        meta.tensor_name = tensor_name ? tensor_name : "tensor";
        meta.shape = T->shape();
        meta.storage_dtype = T->dtype();
        meta.preferred_device = T->device().type;
        meta.quant_scheme = static_cast<mecan::runtime::QuantScheme>(quant_scheme);
        meta.page_layout.page_bytes = (page_bytes == 0) ? (1ULL << 20) : page_bytes;
        meta.total_nbytes = T->numel() * mecan::core::element_size(T->dtype());
        mecan::io::save_paged_mt(*T, meta, filepath ? filepath : "model.mt");
    }

MECAN_EXPORT void* mt_load_paged_mt(const char* filepath) {
        mecan::runtime::PagedTensorMetadata meta{};
        mecan::Tensor loaded = mecan::io::load_paged_mt(filepath ? filepath : "model.mt", &meta);
        return static_cast<void*>(new mecan::Tensor(std::move(loaded)));
    }

    // Advanced 100B Scale Operations
MECAN_EXPORT void mt_flash_ternary_attention(void* q, void* k_cache, void* v_cache, void* out) {
        mecan::Tensor* Q = static_cast<mecan::Tensor*>(q);
        mecan::Tensor* K = static_cast<mecan::Tensor*>(k_cache);
        mecan::Tensor* V = static_cast<mecan::Tensor*>(v_cache);
        mecan::Tensor* OUT = static_cast<mecan::Tensor*>(out);
        mecan::ops::flash_ternary_attention(*Q, *K, *V, *OUT);
    }
    
MECAN_EXPORT void mt_op_bitlinear(void* input, void* weight, void* bias, void* out) {
        mecan::Tensor* IN = static_cast<mecan::Tensor*>(input);
        mecan::Tensor* W = static_cast<mecan::Tensor*>(weight);
        mecan::Tensor* B = static_cast<mecan::Tensor*>(bias);
        mecan::Tensor* OUT = static_cast<mecan::Tensor*>(out);
        mecan::ops::bitlinear_forward(*IN, *W, *B, *OUT);
    }
MECAN_EXPORT void mt_op_qsbits(void* input, void* weight, void* out) {
        mecan::Tensor* IN = static_cast<mecan::Tensor*>(input);
        mecan::Tensor* W = static_cast<mecan::Tensor*>(weight);
        mecan::Tensor* OUT = static_cast<mecan::Tensor*>(out);
        mecan::qsbits::qsbits_forward(*IN, *W, *OUT);
    }

MECAN_EXPORT void mt_op_qsbits_scaled(void* input, void* weight, void* scales, float input_scale, int group_size, void* out) {
        mecan::Tensor* IN = static_cast<mecan::Tensor*>(input);
        mecan::Tensor* W = static_cast<mecan::Tensor*>(weight);
        mecan::Tensor* S = static_cast<mecan::Tensor*>(scales);
        mecan::Tensor* OUT = static_cast<mecan::Tensor*>(out);
        mecan::qsbits::qsbits_forward_scaled(*IN, *W, *S, input_scale, group_size, *OUT);
    }



MECAN_EXPORT void mt_op_max_pool2d(void* input, void* output, int kH, int kW, int stride, int pad) {
    mecan::Tensor* IN = static_cast<mecan::Tensor*>(input);
    mecan::Tensor* OUT = static_cast<mecan::Tensor*>(output);
    auto s = IN->shape();
    mecan::ops::max_pool2d(IN->data_ptr<float>(), OUT->data_ptr<float>(), s[0], s[1], s[2], s[3], kH, kW, stride, pad);
}

MECAN_EXPORT void mt_op_avg_pool2d(void* input, void* output, int kH, int kW, int stride, int pad) {
    mecan::Tensor* IN = static_cast<mecan::Tensor*>(input);
    mecan::Tensor* OUT = static_cast<mecan::Tensor*>(output);
    auto s = IN->shape();
    mecan::ops::avg_pool2d(IN->data_ptr<float>(), OUT->data_ptr<float>(), s[0], s[1], s[2], s[3], kH, kW, stride, pad);
}

MECAN_EXPORT void mt_op_pixel_shuffle(void* input, void* output, int upscale_factor) {
    mecan::Tensor* IN = static_cast<mecan::Tensor*>(input);
    mecan::Tensor* OUT = static_cast<mecan::Tensor*>(output);
    auto s = IN->shape();
    mecan::ops::pixel_shuffle(IN->data_ptr<float>(), OUT->data_ptr<float>(), s[0], s[1], s[2], s[3], upscale_factor);
}

MECAN_EXPORT void mt_vision_canny(void* input, void* output, float low, float high) {
    mecan::Tensor* IN = static_cast<mecan::Tensor*>(input);
    mecan::Tensor* OUT = static_cast<mecan::Tensor*>(output);
    auto s = IN->shape();
    mecan::vision::detect::canny(IN->data_ptr<float>(), OUT->data_ptr<float>(), s[0], s[1], low, high);
}


MECAN_EXPORT void mt_op_pad2d(void* input, void* output, int pad_left, int pad_right, int pad_top, int pad_bottom, int mode, float constant_value) {
    mecan::Tensor* IN = static_cast<mecan::Tensor*>(input);
    mecan::Tensor* OUT = static_cast<mecan::Tensor*>(output);
    auto s = IN->shape();
    mecan::ops::pad2d(IN->data_ptr<float>(), OUT->data_ptr<float>(), s[0], s[1], s[2], s[3], pad_left, pad_right, pad_top, pad_bottom, mode, constant_value);
}

MECAN_EXPORT void mt_op_conv1d(void* input, void* filter, void* output, int stride, int pad) {
    mecan::Tensor* IN = static_cast<mecan::Tensor*>(input);
    mecan::Tensor* F = static_cast<mecan::Tensor*>(filter);
    mecan::Tensor* OUT = static_cast<mecan::Tensor*>(output);
    auto s_in = IN->shape();
    auto s_f = F->shape();
    mecan::ops::conv1d(IN->data_ptr<float>(), s_in[0], s_in[1], s_in[2], F->data_ptr<float>(), s_f[0], s_f[2], OUT->data_ptr<float>(), stride, pad);
}

// ───────────────────────────────────────────────────────────────────────
// ───────────────────────────────────────────────────────────────────────

MECAN_EXPORT void mt_vision_farneback_flow(void* img1, void* img2, void* out_flow, int window_size, int iterations) {
    mecan::Tensor* I1 = static_cast<mecan::Tensor*>(img1);
    mecan::Tensor* I2 = static_cast<mecan::Tensor*>(img2);
    mecan::Tensor* OUT = static_cast<mecan::Tensor*>(out_flow);
    auto s = I1->shape();
    mecan::vision::motion::farneback_flow(I1->data_ptr<float>(), I2->data_ptr<float>(), OUT->data_ptr<float>(), s[0], s[1], window_size, iterations);
}

MECAN_EXPORT void mt_vision_motion_amplify(void* curr_frame, void* bg_avg, void* out_frame, float amp_factor) {
    mecan::Tensor* CF = static_cast<mecan::Tensor*>(curr_frame);
    mecan::Tensor* BG = static_cast<mecan::Tensor*>(bg_avg);
    mecan::Tensor* OUT = static_cast<mecan::Tensor*>(out_frame);
    auto s = CF->shape();
    mecan::vision::motion::motion_amplify(CF->data_ptr<float>(), BG->data_ptr<float>(), OUT->data_ptr<float>(), s[0], s[1], amp_factor);
}

MECAN_EXPORT void mt_vision_background_subtract(void* curr_frame, void* bg_model, void* fg_mask, float lr, float threshold) {
    mecan::Tensor* CF = static_cast<mecan::Tensor*>(curr_frame);
    mecan::Tensor* BG = static_cast<mecan::Tensor*>(bg_model);
    mecan::Tensor* MASK = static_cast<mecan::Tensor*>(fg_mask);
    auto s = CF->shape();
    mecan::vision::motion::background_subtract(CF->data_ptr<float>(), BG->data_ptr<float>(), MASK->data_ptr<float>(), s[0], s[1], lr, threshold);
}

// ───────────────────────────────────────────────────────────────────────
// ───────────────────────────────────────────────────────────────────────

MECAN_EXPORT void mt_vision_signed_distance_field(void* binary_mask, void* out_sdf) {
    mecan::Tensor* M = static_cast<mecan::Tensor*>(binary_mask);
    mecan::Tensor* OUT = static_cast<mecan::Tensor*>(out_sdf);
    auto s = M->shape();
    mecan::vision::vector::signed_distance_field(M->data_ptr<float>(), OUT->data_ptr<float>(), s[0], s[1]);
}

// ───────────────────────────────────────────────────────────────────────
// ───────────────────────────────────────────────────────────────────────

MECAN_EXPORT void mt_vision_illuminance_map(void* img_lum, void* out_lux, float exp_sec, float iso, float f_stop) {
    mecan::Tensor* L = static_cast<mecan::Tensor*>(img_lum);
    mecan::Tensor* OUT = static_cast<mecan::Tensor*>(out_lux);
    mecan::vision::light::illuminance_map(L->data_ptr<float>(), OUT->data_ptr<float>(), L->numel(), exp_sec, iso, f_stop);
}

MECAN_EXPORT void mt_vision_noise_model(void* pristine, void* noisy, float rn, float dc, float exp_sec, float gain) {
    mecan::Tensor* P = static_cast<mecan::Tensor*>(pristine);
    mecan::Tensor* N = static_cast<mecan::Tensor*>(noisy);
    mecan::vision::light::noise_model(P->data_ptr<float>(), N->data_ptr<float>(), P->numel(), rn, dc, exp_sec, gain);
}

MECAN_EXPORT void mt_vision_diffraction_pattern(void* in_img, void* out_img, float aperture_mm, float wave_nm, float focal_mm, float pitch_um) {
    mecan::Tensor* IN = static_cast<mecan::Tensor*>(in_img);
    mecan::Tensor* OUT = static_cast<mecan::Tensor*>(out_img);
    auto s = IN->shape();
    mecan::vision::light::diffraction_pattern(IN->data_ptr<float>(), OUT->data_ptr<float>(), s[0], s[1], aperture_mm, wave_nm, focal_mm, pitch_um);
}

// ───────────────────────────────────────────────────────────────────────
// ───────────────────────────────────────────────────────────────────────

MECAN_EXPORT void mt_color_rgb_to_hsv(void* rgb, void* hsv) {
    mecan::Tensor* IN = static_cast<mecan::Tensor*>(rgb);
    mecan::Tensor* OUT = static_cast<mecan::Tensor*>(hsv);
    mecan::vision::color::rgb_to_hsv(IN->data_ptr<float>(), OUT->data_ptr<float>(), IN->numel()/3);
}

MECAN_EXPORT void mt_color_rgb_to_lab(void* rgb, void* lab) {
    mecan::Tensor* IN = static_cast<mecan::Tensor*>(rgb);
    mecan::Tensor* OUT = static_cast<mecan::Tensor*>(lab);
    mecan::vision::color::rgb_to_lab(IN->data_ptr<float>(), OUT->data_ptr<float>(), IN->numel()/3);
}

MECAN_EXPORT void mt_color_rgb_to_xyz(void* rgb, void* xyz) {
    mecan::Tensor* IN = static_cast<mecan::Tensor*>(rgb);
    mecan::Tensor* OUT = static_cast<mecan::Tensor*>(xyz);
    mecan::vision::color::rgb_to_xyz(IN->data_ptr<float>(), OUT->data_ptr<float>(), IN->numel()/3);
}

MECAN_EXPORT void mt_color_rgb_to_grayscale(void* rgb, void* gray) {
    mecan::Tensor* IN = static_cast<mecan::Tensor*>(rgb);
    mecan::Tensor* OUT = static_cast<mecan::Tensor*>(gray);
    mecan::vision::color::rgb_to_grayscale(IN->data_ptr<float>(), OUT->data_ptr<float>(), IN->numel()/3);
}

MECAN_EXPORT void mt_color_wavelength_to_rgb(float wavelength_nm, void* rgb_out) {
    mecan::Tensor* OUT = static_cast<mecan::Tensor*>(rgb_out);
    mecan::vision::color::wavelength_to_rgb(wavelength_nm, OUT->data_ptr<float>());
}

MECAN_EXPORT void mt_color_dominant(void* rgb, int num_pixels, int k, void* palette, int max_iters) {
    mecan::Tensor* IN = static_cast<mecan::Tensor*>(rgb);
    mecan::Tensor* PAL = static_cast<mecan::Tensor*>(palette);
    mecan::vision::color::dominant_colors(IN->data_ptr<float>(), num_pixels, k, PAL->data_ptr<float>(), max_iters);
}

MECAN_EXPORT void mt_color_hdr_tonemap(void* hdr, void* ldr, int method) {
    mecan::Tensor* IN = static_cast<mecan::Tensor*>(hdr);
    mecan::Tensor* OUT = static_cast<mecan::Tensor*>(ldr);
    mecan::vision::color::hdr_tonemap(IN->data_ptr<float>(), OUT->data_ptr<float>(), IN->numel()/3, method);
}

MECAN_EXPORT void mt_color_rgb_to_spectral(void* rgb, void* spectral) {
    mecan::Tensor* IN = static_cast<mecan::Tensor*>(rgb);
    mecan::Tensor* OUT = static_cast<mecan::Tensor*>(spectral);
    mecan::vision::color::rgb_to_spectral(IN->data_ptr<float>(), OUT->data_ptr<float>(), IN->numel()/3);
}

// ───────────────────────────────────────────────────────────────────────
// ───────────────────────────────────────────────────────────────────────



MECAN_EXPORT float mt_object_edge_density(void* edge_map) {
    mecan::Tensor* E = static_cast<mecan::Tensor*>(edge_map);
    auto s = E->shape();
    return mecan::vision::object::edge_density(E->data_ptr<float>(), s[0], s[1]);
}

MECAN_EXPORT float mt_object_shape_complexity(void* gray) {
    mecan::Tensor* G = static_cast<mecan::Tensor*>(gray);
    auto s = G->shape();
    return mecan::vision::object::shape_complexity(G->data_ptr<float>(), s[0], s[1]);
}

MECAN_EXPORT float mt_object_texture_energy(void* gray) {
    mecan::Tensor* G = static_cast<mecan::Tensor*>(gray);
    auto s = G->shape();
    return mecan::vision::object::texture_energy(G->data_ptr<float>(), s[0], s[1]);
}

MECAN_EXPORT void mt_object_lbp(void* gray, void* out) {
    mecan::Tensor* G = static_cast<mecan::Tensor*>(gray);
    mecan::Tensor* O = static_cast<mecan::Tensor*>(out);
    auto s = G->shape();
    mecan::vision::object::lbp(G->data_ptr<float>(), O->data_ptr<float>(), s[0], s[1]);
}

MECAN_EXPORT void mt_object_bilateral(void* in, void* out, int radius, float sigma_s, float sigma_c) {
    mecan::Tensor* I = static_cast<mecan::Tensor*>(in);
    mecan::Tensor* O = static_cast<mecan::Tensor*>(out);
    auto s = I->shape();
    mecan::vision::object::bilateral_filter(I->data_ptr<float>(), O->data_ptr<float>(), s[0], s[1], radius, sigma_s, sigma_c);
}

MECAN_EXPORT void mt_object_morph_erode(void* in, void* out, int radius) {
    mecan::Tensor* I = static_cast<mecan::Tensor*>(in);
    mecan::Tensor* O = static_cast<mecan::Tensor*>(out);
    auto s = I->shape();
    mecan::vision::object::morph_erode(I->data_ptr<float>(), O->data_ptr<float>(), s[0], s[1], radius);
}

MECAN_EXPORT void mt_object_morph_dilate(void* in, void* out, int radius) {
    mecan::Tensor* I = static_cast<mecan::Tensor*>(in);
    mecan::Tensor* O = static_cast<mecan::Tensor*>(out);
    auto s = I->shape();
    mecan::vision::object::morph_dilate(I->data_ptr<float>(), O->data_ptr<float>(), s[0], s[1], radius);
}

MECAN_EXPORT void mt_object_gradient_map(void* gray, void* mag, void* dir) {
    mecan::Tensor* G = static_cast<mecan::Tensor*>(gray);
    mecan::Tensor* M = static_cast<mecan::Tensor*>(mag);
    mecan::Tensor* D = static_cast<mecan::Tensor*>(dir);
    auto s = G->shape();
    mecan::vision::object::gradient_map(G->data_ptr<float>(), M->data_ptr<float>(), D->data_ptr<float>(), s[0], s[1]);
}

// ───────────────────────────────────────────────────────────────────────
// ───────────────────────────────────────────────────────────────────────

MECAN_EXPORT float mt_temporal_scene_change(void* f1, void* f2) {
    mecan::Tensor* F1 = static_cast<mecan::Tensor*>(f1);
    mecan::Tensor* F2 = static_cast<mecan::Tensor*>(f2);
    auto s = F1->shape();
    return mecan::vision::temporal::scene_change_score(F1->data_ptr<float>(), F2->data_ptr<float>(), s[0], s[1]);
}

MECAN_EXPORT float mt_temporal_stability(void* f1, void* f2) {
    mecan::Tensor* F1 = static_cast<mecan::Tensor*>(f1);
    mecan::Tensor* F2 = static_cast<mecan::Tensor*>(f2);
    auto s = F1->shape();
    return mecan::vision::temporal::frame_stability(F1->data_ptr<float>(), F2->data_ptr<float>(), s[0], s[1]);
}

MECAN_EXPORT void mt_temporal_heatmap(void* f1, void* f2, void* heatmap, float decay) {
    mecan::Tensor* F1 = static_cast<mecan::Tensor*>(f1);
    mecan::Tensor* F2 = static_cast<mecan::Tensor*>(f2);
    mecan::Tensor* HM = static_cast<mecan::Tensor*>(heatmap);
    auto s = F1->shape();
    mecan::vision::temporal::motion_heatmap(F1->data_ptr<float>(), F2->data_ptr<float>(), HM->data_ptr<float>(), s[0], s[1], decay);
}

MECAN_EXPORT void mt_quantum_phase_conv(void* input, void* output, float phase_angle) {
    mecan::Tensor* IN = static_cast<mecan::Tensor*>(input);
    mecan::Tensor* OUT = static_cast<mecan::Tensor*>(output);
    mecan::vision::QuantumVisionModule::quantum_phase_conv(*IN, *OUT, phase_angle);
}

// ═══════════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════════

MECAN_EXPORT void mt_vision_sobel(void* input, void* output, int axis) {
    mecan::Tensor* I = static_cast<mecan::Tensor*>(input);
    mecan::Tensor* O = static_cast<mecan::Tensor*>(output);
    auto s = I->shape();
    mecan::vision::detect::sobel(I->data_ptr<float>(), O->data_ptr<float>(), s[0], s[1], axis);
}

MECAN_EXPORT void mt_vision_gaussian_blur(void* input, void* output, int kernel_size, float sigma) {
    mecan::Tensor* I = static_cast<mecan::Tensor*>(input);
    mecan::Tensor* O = static_cast<mecan::Tensor*>(output);
    auto s = I->shape();
    mecan::vision::detect::gaussian_blur(I->data_ptr<float>(), O->data_ptr<float>(), s[0], s[1], kernel_size, sigma);
}

MECAN_EXPORT void mt_vision_threshold(void* input, void* output, float thresh) {
    // Simple binary threshold on grayscale tensor
    mecan::Tensor* I = static_cast<mecan::Tensor*>(input);
    mecan::Tensor* O = static_cast<mecan::Tensor*>(output);
    size_t n = I->numel();
    const float* ip = I->data_ptr<float>();
    float* op = O->data_ptr<float>();
    #pragma omp parallel for
    for (int64_t i = 0; i < (int64_t)n; ++i) {
        op[i] = ip[i] >= thresh ? 1.0f : 0.0f;
    }
}

MECAN_EXPORT void mt_vision_harris_corners(void* input, void* output, float k, int block_size) {
    mecan::Tensor* I = static_cast<mecan::Tensor*>(input);
    mecan::Tensor* O = static_cast<mecan::Tensor*>(output);
    auto s = I->shape();
    mecan::vision::detect::harris_corners(I->data_ptr<float>(), O->data_ptr<float>(), s[0], s[1], k, block_size);
}

MECAN_EXPORT void mt_vision_resize(void* input, void* output, int new_h, int new_w) {
    // Bilinear interpolation resize
    mecan::Tensor* I = static_cast<mecan::Tensor*>(input);
    mecan::Tensor* O = static_cast<mecan::Tensor*>(output);
    auto s = I->shape();
    size_t old_h = s[0], old_w = s[1];
    const float* ip = I->data_ptr<float>();
    float* op = O->data_ptr<float>();
    float ry = (float)old_h / (float)new_h;
    float rx = (float)old_w / (float)new_w;
    #pragma omp parallel for
    for (int y = 0; y < new_h; ++y) {
        for (int x = 0; x < new_w; ++x) {
            float sy = y * ry, sx = x * rx;
            int y0 = (int)sy, x0 = (int)sx;
            int y1 = y0 + 1 < (int)old_h ? y0 + 1 : y0;
            int x1 = x0 + 1 < (int)old_w ? x0 + 1 : x0;
            float fy = sy - y0, fx = sx - x0;
            float v00 = ip[y0 * old_w + x0];
            float v01 = ip[y0 * old_w + x1];
            float v10 = ip[y1 * old_w + x0];
            float v11 = ip[y1 * old_w + x1];
            op[y * new_w + x] = v00*(1-fy)*(1-fx) + v01*(1-fy)*fx + v10*fy*(1-fx) + v11*fy*fx;
        }
    }
}

MECAN_EXPORT void mt_vision_rotate(void* input, void* output, float angle_deg) {
    // Rotation via inverse mapping with bilinear interpolation
    mecan::Tensor* I = static_cast<mecan::Tensor*>(input);
    mecan::Tensor* O = static_cast<mecan::Tensor*>(output);
    auto s = I->shape();
    size_t H = s[0], W = s[1];
    const float* ip = I->data_ptr<float>();
    float* op = O->data_ptr<float>();
    float rad = angle_deg * 3.14159265f / 180.0f;
    float cos_a = std::cos(rad), sin_a = std::sin(rad);
    float cx = W * 0.5f, cy = H * 0.5f;
    #pragma omp parallel for
    for (int y = 0; y < (int)H; ++y) {
        for (int x = 0; x < (int)W; ++x) {
            float dx = x - cx, dy = y - cy;
            float sx = cos_a * dx + sin_a * dy + cx;
            float sy = -sin_a * dx + cos_a * dy + cy;
            if (sx >= 0 && sx < W - 1 && sy >= 0 && sy < H - 1) {
                int x0 = (int)sx, y0 = (int)sy;
                float fx = sx - x0, fy = sy - y0;
                op[y*W+x] = ip[y0*W+x0]*(1-fy)*(1-fx) + ip[y0*W+x0+1]*(1-fy)*fx
                           + ip[(y0+1)*W+x0]*fy*(1-fx) + ip[(y0+1)*W+x0+1]*fy*fx;
            } else {
                op[y*W+x] = 0.0f;
            }
        }
    }
}

MECAN_EXPORT void mt_vision_histogram(void* input, void* output, int bins) {
    // Compute normalized histogram of a grayscale tensor
    mecan::Tensor* I = static_cast<mecan::Tensor*>(input);
    mecan::Tensor* O = static_cast<mecan::Tensor*>(output);
    size_t n = I->numel();
    const float* ip = I->data_ptr<float>();
    float* op = O->data_ptr<float>();
    for (int i = 0; i < bins; ++i) op[i] = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        int bin = (int)(ip[i] * (bins - 1));
        if (bin < 0) bin = 0;
        if (bin >= bins) bin = bins - 1;
        op[bin] += 1.0f;
    }
    float inv_n = 1.0f / (float)n;
    for (int i = 0; i < bins; ++i) op[i] *= inv_n;
}

MECAN_EXPORT int mt_vision_connected_components(void* input, void* output) {
    // Simple single-pass connected components labeling (4-connectivity)
    mecan::Tensor* I = static_cast<mecan::Tensor*>(input);
    mecan::Tensor* O = static_cast<mecan::Tensor*>(output);
    auto s = I->shape();
    size_t H = s[0], W = s[1];
    const float* ip = I->data_ptr<float>();
    float* op = O->data_ptr<float>();
    for (size_t i = 0; i < H * W; ++i) op[i] = 0.0f;
    int label = 0;
    // Two-pass simplified
    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            if (ip[y*W+x] <= 0.5f || op[y*W+x] > 0.0f) continue;
            label++;
            // BFS flood fill
            std::vector<std::pair<int,int>> stack;
            stack.push_back({(int)y, (int)x});
            while (!stack.empty()) {
                std::pair<int,int> pos = stack.back(); stack.pop_back();
                int cy2 = pos.first, cx2 = pos.second;
                if (cy2 < 0 || cy2 >= (int)H || cx2 < 0 || cx2 >= (int)W) continue;
                if (ip[cy2*W+cx2] <= 0.5f || op[cy2*W+cx2] > 0.0f) continue;
                op[cy2*W+cx2] = (float)label;
                stack.push_back({cy2-1, cx2});
                stack.push_back({cy2+1, cx2});
                stack.push_back({cy2, cx2-1});
                stack.push_back({cy2, cx2+1});
            }
        }
    }
    return label;
}

MECAN_EXPORT void mt_vision_nms(void* boxes, void* scores, void* keep, int num_boxes, float iou_threshold) {
    // Non-Maximum Suppression for detection boxes
    // boxes: [num_boxes, 4] (x1,y1,x2,y2), scores: [num_boxes], keep: [num_boxes] (0 or 1)
    float* b = static_cast<mecan::Tensor*>(boxes)->data_ptr<float>();
    float* sc = static_cast<mecan::Tensor*>(scores)->data_ptr<float>();
    float* kp = static_cast<mecan::Tensor*>(keep)->data_ptr<float>();
    // Sort indices by score (selection sort for simplicity)
    std::vector<int> idx(num_boxes);
    for (int i = 0; i < num_boxes; ++i) { idx[i] = i; kp[i] = 1.0f; }
    for (int i = 0; i < num_boxes - 1; ++i)
        for (int j = i + 1; j < num_boxes; ++j)
            if (sc[idx[j]] > sc[idx[i]]) std::swap(idx[i], idx[j]);
    for (int i = 0; i < num_boxes; ++i) {
        int ii = idx[i];
        if (kp[ii] < 0.5f) continue;
        for (int j = i + 1; j < num_boxes; ++j) {
            int jj = idx[j];
            if (kp[jj] < 0.5f) continue;
            // Compute IoU
            float x1 = std::max(b[ii*4], b[jj*4]);
            float y1 = std::max(b[ii*4+1], b[jj*4+1]);
            float x2 = std::min(b[ii*4+2], b[jj*4+2]);
            float y2 = std::min(b[ii*4+3], b[jj*4+3]);
            float inter = std::max(0.0f, x2-x1) * std::max(0.0f, y2-y1);
            float a1 = (b[ii*4+2]-b[ii*4]) * (b[ii*4+3]-b[ii*4+1]);
            float a2 = (b[jj*4+2]-b[jj*4]) * (b[jj*4+3]-b[jj*4+1]);
            float iou = inter / (a1 + a2 - inter + 1e-6f);
            if (iou > iou_threshold) kp[jj] = 0.0f;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════════

MECAN_EXPORT void mt_vision_lucas_kanade(void* img1, void* img2,
    void* pts_x, void* pts_y, void* out_u, void* out_v, int num_pts, int window_size) {
    mecan::Tensor* I1 = static_cast<mecan::Tensor*>(img1);
    mecan::Tensor* I2 = static_cast<mecan::Tensor*>(img2);
    auto s = I1->shape();
    mecan::vision::motion::lucas_kanade_flow(
        I1->data_ptr<float>(), I2->data_ptr<float>(),
        static_cast<mecan::Tensor*>(pts_x)->data_ptr<float>(),
        static_cast<mecan::Tensor*>(pts_y)->data_ptr<float>(),
        static_cast<mecan::Tensor*>(out_u)->data_ptr<float>(),
        static_cast<mecan::Tensor*>(out_v)->data_ptr<float>(),
        (size_t)num_pts, s[0], s[1], window_size
    );
}

MECAN_EXPORT void mt_vision_bg_subtract(void* current, void* bg_model, void* fg_mask,
    float learning_rate, float threshold) {
    mecan::Tensor* C = static_cast<mecan::Tensor*>(current);
    mecan::Tensor* BG = static_cast<mecan::Tensor*>(bg_model);
    mecan::Tensor* FG = static_cast<mecan::Tensor*>(fg_mask);
    auto s = C->shape();
    mecan::vision::motion::background_subtract(
        C->data_ptr<float>(), BG->data_ptr<float>(), FG->data_ptr<float>(),
        s[0], s[1], learning_rate, threshold
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════════

MECAN_EXPORT void mt_vision_spectral_response(void* input, void* output) {
    // luminance extraction (physical spectral response)
    mecan::Tensor* I = static_cast<mecan::Tensor*>(input);
    mecan::Tensor* O = static_cast<mecan::Tensor*>(output);
    size_t n = I->numel() / 3; // RGB → luminance
    mecan::vision::light::luminance(I->data_ptr<float>(), O->data_ptr<float>(), n);
}

MECAN_EXPORT void mt_vision_photon_noise(void* pristine, void* noisy,
    float read_noise, float dark_current, float exposure, float gain) {
    mecan::Tensor* P = static_cast<mecan::Tensor*>(pristine);
    mecan::Tensor* N = static_cast<mecan::Tensor*>(noisy);
    mecan::vision::light::noise_model(
        P->data_ptr<float>(), N->data_ptr<float>(), P->numel(),
        read_noise, dark_current, exposure, gain
    );
}

MECAN_EXPORT void mt_vision_radiance(void* luminance, void* illuminance,
    float exposure_time, float iso, float aperture) {
    mecan::Tensor* L = static_cast<mecan::Tensor*>(luminance);
    mecan::Tensor* I = static_cast<mecan::Tensor*>(illuminance);
    mecan::vision::light::illuminance_map(
        L->data_ptr<float>(), I->data_ptr<float>(), L->numel(),
        exposure_time, iso, aperture
    );
}

MECAN_EXPORT void mt_vision_diffraction(void* input, void* output,
    float aperture_mm, float wavelength_nm, float focal_mm, float pixel_pitch_um) {
    mecan::Tensor* I = static_cast<mecan::Tensor*>(input);
    mecan::Tensor* O = static_cast<mecan::Tensor*>(output);
    auto s = I->shape();
    mecan::vision::light::diffraction_pattern(
        I->data_ptr<float>(), O->data_ptr<float>(), s[0], s[1],
        aperture_mm, wavelength_nm, focal_mm, pixel_pitch_um
    );
}

MECAN_EXPORT void mt_vision_ray_refract(void* input, void* output, float gamma) {
    // gamma correction (physical opto-electronic transfer)
    mecan::Tensor* I = static_cast<mecan::Tensor*>(input);
    mecan::Tensor* O = static_cast<mecan::Tensor*>(output);
    mecan::vision::light::gamma_correct(I->data_ptr<float>(), O->data_ptr<float>(), I->numel(), gamma);
}

#ifdef _WIN32
    MECAN_EXPORT void* PyInit__core() { return nullptr; }
#endif

}
