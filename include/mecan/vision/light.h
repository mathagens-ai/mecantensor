#pragma once

#include "mecan/tensor.h"
#include <cstddef>

namespace mecan {
namespace vision {
namespace light {

    /**
     * Physical Luminance Extraction
     * Converts sRGB directly to relative luminance (cd/m^2 approximation)
     */
    void luminance(const float* img_rgb, float* out_lum, size_t num_pixels);

    /**
     * Illuminance Map
     * Maps physical pixel brightness to estimated Lux values based on exposure parameters.
     */
    void illuminance_map(
        const float* img_lum, float* out_lux, size_t num_pixels,
        float exposure_time_sec, float iso, float aperture_f_stop
    );

    /**
     * Photon Counting Simulation
     * Converts continuous intensity to discrete photon arrival counts (Poisson process simulation).
     */
    void photon_count(
        const float* img_lum, float* out_photons, size_t num_pixels,
        float quantum_efficiency, float pixel_area_um2, float exposure_sec
    );

    /**
     * Advanced Sensor Noise Model
     * Injects realistic read noise, dark current noise, and shot noise into a pristine image.
     */
    void noise_model(
        const float* pristine_img, float* noisy_img, size_t num_pixels,
        float read_noise_e, float dark_current_e_per_sec, float exposure_sec, float gain
    );

    /**
     * Diffraction Pattern (Airy Disk Convolution Approximation)
     * Simulates the blurring caused by light diffracting through a circular aperture.
     */
    void diffraction_pattern(
        const float* in_img, float* out_img, size_t H, size_t W,
        float aperture_diameter_mm, float wavelength_nm, float focal_length_mm, float pixel_pitch_um
    );

    /**
     * Linear Polarization Filter
     * Simulates a rotating polarizing filter over an image given an angle.
     * (Assumes input contains unpolarized and polarized channels, simplified here to intensity modulation).
     */
    void polarization_filter(
        const float* img_intensity, const float* img_angle_of_polarization, const float* degree_of_polarization,
        float* out_img, size_t num_pixels, float filter_angle_rad
    );

    /**
     * HDR Exposure Merge (Mertens Fusion Approximation)
     * Fuses multiple exposures of the same scene into a single high dynamic range image.
     */
    void exposure_merge(
        const float** img_stack, size_t num_exposures,
        float* out_hdr, size_t num_pixels
    );

    /**
     * Depth From Focus
     * Estimates a depth map by analyzing local high-frequency contrast across a focal stack.
     */
    void depth_from_focus(
        const float** img_stack, const float* focus_distances, size_t num_exposures,
        float* out_depth_map, size_t H, size_t W
    );

    /**
     * Wave Decomposition (Spatial Frequency)
     * Extracts a specific spatial frequency band (Laplacian approximation).
     */
    void wave_decompose(
        const float* img, float* out_band, size_t H, size_t W,
        float sigma_low, float sigma_high
    );

    /**
     * Physical Gamma Correction
     * Applies precise opto-electronic transfer functions.
     */
    void gamma_correct(
        const float* linear_img, float* gamma_img, size_t num_pixels,
        float gamma = 2.2f
    );

} // namespace light
} // namespace vision
} // namespace mecan
