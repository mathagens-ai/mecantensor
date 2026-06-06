#include "mecan/vision/light.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <omp.h>

namespace mecan {
namespace vision {
namespace light {

    void luminance(const float* img_rgb, float* out_lum, size_t num_pixels) {
        #pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i) {
            float r = img_rgb[i * 3 + 0];
            float g = img_rgb[i * 3 + 1];
            float b = img_rgb[i * 3 + 2];
            out_lum[i] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        }
    }

    void illuminance_map(
        const float* img_lum, float* out_lux, size_t num_pixels,
        float exposure_time_sec, float iso, float aperture_f_stop
    ) {
        float calibration_constant = 250.0f; // Typical camera calibration constant
        float factor = calibration_constant * (aperture_f_stop * aperture_f_stop) / (exposure_time_sec * iso);
        #pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i) {
            out_lux[i] = img_lum[i] * factor;
        }
    }

    void photon_count(
        const float* img_lum, float* out_photons, size_t num_pixels,
        float quantum_efficiency, float pixel_area_um2, float exposure_sec
    ) {
        // Lux to photon rate roughly 5x10^11 photons / (s * m^2 * lux) -> approx conversion
        float area_m2 = pixel_area_um2 * 1e-12f;
        float factor = 5e11f * quantum_efficiency * area_m2 * exposure_sec;
        #pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i) {
            float expected_photons = img_lum[i] * factor; // Mocked expected
            out_photons[i] = expected_photons; // We can add poisson noise in noise_model
        }
    }

    void noise_model(
        const float* pristine_img, float* noisy_img, size_t num_pixels,
        float read_noise_e, float dark_current_e_per_sec, float exposure_sec, float gain
    ) {
        float dark_e = dark_current_e_per_sec * exposure_sec;
        #pragma omp parallel
        {
            std::random_device rd;
            std::mt19937 gen(rd() ^ omp_get_thread_num());
            std::normal_distribution<float> read_dist(0.0f, read_noise_e);

            #pragma omp for
            for (int i = 0; i < (int)num_pixels; ++i) {
                float signal_e = pristine_img[i] * gain; // Simplified conversion
                std::poisson_distribution<int> shot_dist(signal_e + dark_e);
                float total_e = (float)shot_dist(gen) + read_dist(gen);
                noisy_img[i] = std::max(0.0f, total_e / gain);
            }
        }
    }

    void diffraction_pattern(
        const float* in_img, float* out_img, size_t H, size_t W,
        float aperture_diameter_mm, float wavelength_nm, float focal_length_mm, float pixel_pitch_um
    ) {
        // Rayleigh criterion approx
        float lambda_m = wavelength_nm * 1e-9f;
        float D_m = aperture_diameter_mm * 1e-3f;
        float f_m = focal_length_mm * 1e-3f;
        float pitch_m = pixel_pitch_um * 1e-6f;
        
        float airy_radius_m = 1.22f * lambda_m * f_m / D_m;
        float airy_radius_px = airy_radius_m / pitch_m;
        
        int kH = std::max(1, (int)(airy_radius_px * 2.0f));
        int kW = kH;
        int half_k = kH / 2;

        #pragma omp parallel for
        for (int y = 0; y < (int)H; ++y) {
            for (int x = 0; x < (int)W; ++x) {
                float sum = 0.0f;
                float weight_sum = 0.0f;
                for (int wy = -half_k; wy <= half_k; ++wy) {
                    for (int wx = -half_k; wx <= half_k; ++wx) {
                        int cy = std::max(0, std::min((int)H - 1, y + wy));
                        int cx = std::max(0, std::min((int)W - 1, x + wx));
                        float r = std::sqrt((float)(wx * wx + wy * wy));
                        if (r <= airy_radius_px) {
                            float w = 1.0f - (r / (airy_radius_px + 1e-5f)); // Simplified airy disk
                            sum += in_img[cy * W + cx] * w;
                            weight_sum += w;
                        }
                    }
                }
                out_img[y * W + x] = weight_sum > 0 ? sum / weight_sum : in_img[y * W + x];
            }
        }
    }

    void polarization_filter(
        const float* img_intensity, const float* img_angle_of_polarization, const float* degree_of_polarization,
        float* out_img, size_t num_pixels, float filter_angle_rad
    ) {
        #pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i) {
            float I = img_intensity[i];
            float phi = img_angle_of_polarization[i];
            float rho = degree_of_polarization[i];
            // Malus's Law approx
            float transmitted = I * 0.5f * (1.0f + rho * std::cos(2.0f * (phi - filter_angle_rad)));
            out_img[i] = transmitted;
        }
    }

    void exposure_merge(
        const float** img_stack, size_t num_exposures,
        float* out_hdr, size_t num_pixels
    ) {
        #pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i) {
            float sum_w = 0.0f;
            float sum_img = 0.0f;
            for (int j = 0; j < (int)num_exposures; ++j) {
                float val = img_stack[j][i];
                // Weight based on exposed well capacity (val closer to 0.5 is better)
                float w = std::exp(-12.5f * (val - 0.5f) * (val - 0.5f));
                sum_w += w;
                sum_img += val * w;
            }
            out_hdr[i] = sum_w > 0 ? sum_img / sum_w : 0.0f;
        }
    }

    void depth_from_focus(
        const float** img_stack, const float* focus_distances, size_t num_exposures,
        float* out_depth_map, size_t H, size_t W
    ) {
        #pragma omp parallel for
        for (int y = 0; y < (int)H; ++y) {
            for (int x = 0; x < (int)W; ++x) {
                float max_focus_measure = -1e9f;
                float best_depth = 0.0f;
                int idx = y * W + x;
                
                for (int j = 0; j < (int)num_exposures; ++j) {
                    float val = img_stack[j][idx];
                    // Approximate focus measure using center pixel value (needs proper laplacian in real usage)
                    float focus_measure = val; 
                    if (focus_measure > max_focus_measure) {
                        max_focus_measure = focus_measure;
                        best_depth = focus_distances[j];
                    }
                }
                out_depth_map[idx] = best_depth;
            }
        }
    }

    void wave_decompose(
        const float* img, float* out_band, size_t H, size_t W,
        float sigma_low, float sigma_high
    ) {
        #pragma omp parallel for
        for (int i = 0; i < (int)(H * W); ++i) {
            out_band[i] = img[i]; // Placeholder for difference of gaussians
        }
    }

    void gamma_correct(
        const float* linear_img, float* gamma_img, size_t num_pixels,
        float gamma
    ) {
        float inv_gamma = 1.0f / gamma;
        #pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i) {
            gamma_img[i] = std::pow(std::max(0.0f, linear_img[i]), inv_gamma);
        }
    }

} // namespace light
} // namespace vision
} // namespace mecan
