#pragma once
// MecanTensor Vision: Spectral Color Engine
// 20 color operations covering 10 color spaces, HDR, spectral wavelengths,
// perceptual distance metrics, and chromatic adaptation.
//
// All functions operate on raw float* arrays in CHW (channel, height, width)
// layout with values in [0,1] for standard images, or unbounded for HDR/Linear.
//
// Supported color spaces:
//   sRGB, Linear RGB, CIE XYZ, CIE L*a*b*, HSV, HSL, LCH, LMS,
//   Grayscale, Spectral (81-band, 380-780nm @ 5nm)

#include <cstddef>
#include <cstdint>

namespace mecan {
namespace vision {
namespace color {

    // ─── Color Space Conversions ─────────────────────────────────────────────

    // 1. sRGB → CIE 1931 XYZ (physics-accurate, D65 illuminant)
    void rgb_to_xyz(const float* rgb, float* xyz, size_t num_pixels);

    // 2. CIE XYZ → sRGB
    void xyz_to_rgb(const float* xyz, float* rgb, size_t num_pixels);

    // 3. sRGB → CIE L*a*b* (perceptually uniform)
    void rgb_to_lab(const float* rgb, float* lab, size_t num_pixels);

    // 4. CIE L*a*b* → sRGB
    void lab_to_rgb(const float* lab, float* rgb, size_t num_pixels);

    // 5. sRGB → HSV (Hue [0-360], Saturation [0-1], Value [0-1])
    void rgb_to_hsv(const float* rgb, float* hsv, size_t num_pixels);

    // 6. HSV → sRGB
    void hsv_to_rgb(const float* hsv, float* rgb, size_t num_pixels);

    // 7. sRGB → HSL (Hue [0-360], Saturation [0-1], Lightness [0-1])
    void rgb_to_hsl(const float* rgb, float* hsl, size_t num_pixels);

    // 8. sRGB → LMS cone response (Long, Medium, Short)
    void rgb_to_lms(const float* rgb, float* lms, size_t num_pixels);

    // 9. sRGB → Linear RGB (remove gamma curve)
    void rgb_to_linear(const float* srgb, float* linear, size_t num_pixels);

    // 10. Linear RGB → sRGB (apply gamma curve)
    void linear_to_rgb(const float* linear, float* srgb, size_t num_pixels);

    // 11. sRGB → Grayscale (BT.709 luminosity weights)
    void rgb_to_grayscale(const float* rgb, float* gray, size_t num_pixels);

    // ─── Spectral Operations ─────────────────────────────────────────────────

    // 12. sRGB → Spectral power distribution (81 bands, 380-780nm @ 5nm)
    void rgb_to_spectral(const float* rgb, float* spectral, size_t num_pixels);

    // 13. Spectral → sRGB (integrate spectral via CIE color matching functions)
    void spectral_to_rgb(const float* spectral, float* rgb, size_t num_pixels);

    // 14. Single wavelength (380-780nm) → visible RGB color
    void wavelength_to_rgb(float lambda_nm, float* rgb);

    // ─── Color Analysis ──────────────────────────────────────────────────────

    // 15. Perceptual color distance (Delta-E)
    //     metric: 0=CIE76, 1=CIE94, 2=CIEDE2000
    float color_distance(const float* lab1, const float* lab2, int metric);

    // 16. Bradford chromatic adaptation (change illuminant)
    //     src_white, dst_white: XYZ of source and destination white points
    void chromatic_adapt(const float* xyz_in, float* xyz_out, size_t num_pixels,
                         const float* src_white, const float* dst_white);

    // 17. HDR tonemapping (compress HDR to displayable range)
    //     method: 0=Reinhard, 1=ACES, 2=Filmic
    void hdr_tonemap(const float* hdr, float* ldr, size_t num_pixels, int method);

    // 18. Color histogram in any color space
    //     Computes histogram for each channel separately
    void color_histogram(const float* img, size_t num_pixels, int num_channels,
                         int num_bins, float* histogram_out);

    // 19. Extract K dominant colors via K-means clustering
    void dominant_colors(const float* rgb, size_t num_pixels, int k,
                         float* palette_out, int max_iters);

    // 20. Quantize image to N-color palette
    void color_quantize(const float* rgb, float* out, size_t num_pixels,
                        const float* palette, int palette_size);

    // ─── Helpers ─────────────────────────────────────────────────────────────

    // Gamma helpers (used internally)
    float srgb_to_linear_scalar(float v);
    float linear_to_srgb_scalar(float v);

    // CIE L*a*b* helpers
    float lab_f(float t);
    float lab_f_inv(float t);

    // D65 standard illuminant white point in XYZ
    constexpr float D65_X = 0.95047f;
    constexpr float D65_Y = 1.00000f;
    constexpr float D65_Z = 1.08883f;

} // namespace color
} // namespace vision
} // namespace mecan
