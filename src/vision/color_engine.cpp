// MecanTensor Vision: Spectral Color Engine — 20 Operations
#include "mecan/vision/color.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>
#include <omp.h>

namespace mecan {
namespace vision {
namespace color {

    // ─── Gamma Helpers ──────────────────────────────────────────────────────
    float srgb_to_linear_scalar(float v) {
        return (v <= 0.04045f) ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f);
    }
    float linear_to_srgb_scalar(float v) {
        return (v <= 0.0031308f) ? v * 12.92f : 1.055f * std::pow(v, 1.0f/2.4f) - 0.055f;
    }

    // ─── Lab Helpers ────────────────────────────────────────────────────────
    float lab_f(float t) {
        const float d = 6.0f/29.0f;
        return (t > d*d*d) ? std::cbrt(t) : t/(3.0f*d*d) + 4.0f/29.0f;
    }
    float lab_f_inv(float t) {
        const float d = 6.0f/29.0f;
        return (t > d) ? t*t*t : 3.0f*d*d*(t - 4.0f/29.0f);
    }

    // ─── 1. sRGB → XYZ ─────────────────────────────────────────────────────
    void rgb_to_xyz(const float* rgb, float* xyz, size_t num_pixels) {
        #pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i) {
            float r = srgb_to_linear_scalar(rgb[i*3+0]);
            float g = srgb_to_linear_scalar(rgb[i*3+1]);
            float b = srgb_to_linear_scalar(rgb[i*3+2]);
            xyz[i*3+0] = 0.4124564f*r + 0.3575761f*g + 0.1804375f*b;
            xyz[i*3+1] = 0.2126729f*r + 0.7151522f*g + 0.0721750f*b;
            xyz[i*3+2] = 0.0193339f*r + 0.1191920f*g + 0.9503041f*b;
        }
    }

    // ─── 2. XYZ → sRGB ─────────────────────────────────────────────────────
    void xyz_to_rgb(const float* xyz, float* rgb, size_t num_pixels) {
        #pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i) {
            float x=xyz[i*3], y=xyz[i*3+1], z=xyz[i*3+2];
            float r =  3.2404542f*x - 1.5371385f*y - 0.4985314f*z;
            float g = -0.9692660f*x + 1.8760108f*y + 0.0415560f*z;
            float b =  0.0556434f*x - 0.2040259f*y + 1.0572252f*z;
            rgb[i*3+0] = linear_to_srgb_scalar(std::max(0.f, r));
            rgb[i*3+1] = linear_to_srgb_scalar(std::max(0.f, g));
            rgb[i*3+2] = linear_to_srgb_scalar(std::max(0.f, b));
        }
    }

    // ─── 3. sRGB → L*a*b* ──────────────────────────────────────────────────
    void rgb_to_lab(const float* rgb, float* lab, size_t num_pixels) {
        #pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i) {
            float r = srgb_to_linear_scalar(rgb[i*3+0]);
            float g = srgb_to_linear_scalar(rgb[i*3+1]);
            float b = srgb_to_linear_scalar(rgb[i*3+2]);
            float x = (0.4124564f*r + 0.3575761f*g + 0.1804375f*b) / D65_X;
            float y = (0.2126729f*r + 0.7151522f*g + 0.0721750f*b) / D65_Y;
            float z = (0.0193339f*r + 0.1191920f*g + 0.9503041f*b) / D65_Z;
            float fx = lab_f(x), fy = lab_f(y), fz = lab_f(z);
            lab[i*3+0] = 116.0f * fy - 16.0f;
            lab[i*3+1] = 500.0f * (fx - fy);
            lab[i*3+2] = 200.0f * (fy - fz);
        }
    }

    // ─── 4. L*a*b* → sRGB ──────────────────────────────────────────────────
    void lab_to_rgb(const float* lab, float* rgb, size_t num_pixels) {
        #pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i) {
            float L=lab[i*3], a=lab[i*3+1], b=lab[i*3+2];
            float fy = (L + 16.0f) / 116.0f;
            float fx = a / 500.0f + fy;
            float fz = fy - b / 200.0f;
            float x = D65_X * lab_f_inv(fx);
            float y = D65_Y * lab_f_inv(fy);
            float z = D65_Z * lab_f_inv(fz);
            float r =  3.2404542f*x - 1.5371385f*y - 0.4985314f*z;
            float g = -0.9692660f*x + 1.8760108f*y + 0.0415560f*z;
            float bl=  0.0556434f*x - 0.2040259f*y + 1.0572252f*z;
            rgb[i*3+0] = linear_to_srgb_scalar(std::max(0.f, r));
            rgb[i*3+1] = linear_to_srgb_scalar(std::max(0.f, g));
            rgb[i*3+2] = linear_to_srgb_scalar(std::max(0.f, bl));
        }
    }

    // ─── 5. sRGB → HSV ─────────────────────────────────────────────────────
    void rgb_to_hsv(const float* rgb, float* hsv, size_t num_pixels) {
        #pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i) {
            float r=rgb[i*3], g=rgb[i*3+1], b=rgb[i*3+2];
            float mx = std::max({r,g,b}), mn = std::min({r,g,b});
            float d = mx - mn;
            float h = 0, s = (mx > 0) ? d/mx : 0;
            if (d > 1e-6f) {
                if (mx == r)      h = 60.f * std::fmod((g-b)/d + 6.f, 6.f);
                else if (mx == g) h = 60.f * ((b-r)/d + 2.f);
                else              h = 60.f * ((r-g)/d + 4.f);
            }
            hsv[i*3+0]=h; hsv[i*3+1]=s; hsv[i*3+2]=mx;
        }
    }

    // ─── 6. HSV → sRGB ─────────────────────────────────────────────────────
    void hsv_to_rgb(const float* hsv, float* rgb, size_t num_pixels) {
        #pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i) {
            float h=hsv[i*3], s=hsv[i*3+1], v=hsv[i*3+2];
            float c=v*s, x=c*(1.f-std::abs(std::fmod(h/60.f,2.f)-1.f)), m=v-c;
            float r=0,g=0,b=0;
            if(h<60){r=c;g=x;} else if(h<120){r=x;g=c;} else if(h<180){g=c;b=x;}
            else if(h<240){g=x;b=c;} else if(h<300){r=x;b=c;} else{r=c;b=x;}
            rgb[i*3+0]=r+m; rgb[i*3+1]=g+m; rgb[i*3+2]=b+m;
        }
    }

    // ─── 7. sRGB → HSL ─────────────────────────────────────────────────────
    void rgb_to_hsl(const float* rgb, float* hsl, size_t num_pixels) {
        #pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i) {
            float r=rgb[i*3], g=rgb[i*3+1], b=rgb[i*3+2];
            float mx=std::max({r,g,b}), mn=std::min({r,g,b});
            float l=(mx+mn)*0.5f, d=mx-mn, h=0, s=0;
            if(d>1e-6f){
                s = (l>0.5f) ? d/(2.f-mx-mn) : d/(mx+mn);
                if(mx==r) h=60.f*std::fmod((g-b)/d+6.f,6.f);
                else if(mx==g) h=60.f*((b-r)/d+2.f);
                else h=60.f*((r-g)/d+4.f);
            }
            hsl[i*3+0]=h; hsl[i*3+1]=s; hsl[i*3+2]=l;
        }
    }

    // ─── 8. sRGB → LMS ─────────────────────────────────────────────────────
    void rgb_to_lms(const float* rgb, float* lms, size_t num_pixels) {
        #pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i) {
            float r=srgb_to_linear_scalar(rgb[i*3]);
            float g=srgb_to_linear_scalar(rgb[i*3+1]);
            float b=srgb_to_linear_scalar(rgb[i*3+2]);
            lms[i*3+0] = 0.3811f*r + 0.5783f*g + 0.0402f*b;
            lms[i*3+1] = 0.1967f*r + 0.7244f*g + 0.0782f*b;
            lms[i*3+2] = 0.0241f*r + 0.1288f*g + 0.8444f*b;
        }
    }

    // ─── 9. sRGB → Linear RGB ──────────────────────────────────────────────
    void rgb_to_linear(const float* srgb, float* linear, size_t num_pixels) {
        #pragma omp parallel for
        for (int i = 0; i < (int)(num_pixels*3); ++i)
            linear[i] = srgb_to_linear_scalar(srgb[i]);
    }

    // ─── 10. Linear → sRGB ─────────────────────────────────────────────────
    void linear_to_rgb(const float* linear, float* srgb, size_t num_pixels) {
        #pragma omp parallel for
        for (int i = 0; i < (int)(num_pixels*3); ++i)
            srgb[i] = linear_to_srgb_scalar(std::max(0.f, linear[i]));
    }

    // ─── 11. sRGB → Grayscale ──────────────────────────────────────────────
    void rgb_to_grayscale(const float* rgb, float* gray, size_t num_pixels) {
        #pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i)
            gray[i] = 0.2126f*rgb[i*3] + 0.7152f*rgb[i*3+1] + 0.0722f*rgb[i*3+2];
    }

    // ─── 12. sRGB → Spectral (81-band) ─────────────────────────────────────
    void rgb_to_spectral(const float* rgb, float* spectral, size_t num_pixels) {
        // Simplified: distribute RGB energy across 81 bands (380-780nm @ 5nm)
        #pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i) {
            float r=rgb[i*3], g=rgb[i*3+1], b=rgb[i*3+2];
            for (int band = 0; band < 81; ++band) {
                float wl = 380.0f + band * 5.0f;
                float w = 0.0f;
                // Approximate CIE color matching
                if (wl >= 380 && wl < 440) w = b * (1.f - (wl-380.f)/60.f) + 0.1f*r;
                else if (wl < 490) w = b * (1.f-(wl-440.f)/50.f) + g*((wl-440.f)/50.f);
                else if (wl < 510) w = g;
                else if (wl < 580) w = g*(1.f-(wl-510.f)/70.f) + r*((wl-510.f)/70.f);
                else if (wl < 645) w = r;
                else if (wl <= 780) w = r * (1.f - (wl-645.f)/135.f);
                spectral[i*81 + band] = std::max(0.f, w);
            }
        }
    }

    // ─── 13. Spectral → sRGB ───────────────────────────────────────────────
    void spectral_to_rgb(const float* spectral, float* rgb, size_t num_pixels) {
        #pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i) {
            float r=0, g=0, b=0;
            for (int band = 0; band < 81; ++band) {
                float wl = 380.0f + band * 5.0f;
                float s = spectral[i*81 + band];
                // Approximate inverse mapping
                if (wl >= 580 && wl < 645) r += s;
                else if (wl >= 645) r += s * 0.5f;
                if (wl >= 490 && wl < 580) g += s;
                if (wl >= 380 && wl < 490) b += s;
            }
            float norm = 1.0f / 81.0f;
            rgb[i*3+0] = std::min(1.f, r*norm*5.f);
            rgb[i*3+1] = std::min(1.f, g*norm*5.f);
            rgb[i*3+2] = std::min(1.f, b*norm*5.f);
        }
    }

    // ─── 14. Wavelength → RGB ──────────────────────────────────────────────
    void wavelength_to_rgb(float wl, float* rgb) {
        float r=0,g=0,b=0;
        if (wl >= 380 && wl < 440) { r = -(wl-440)/(440-380); b = 1.0f; }
        else if (wl < 490) { g = (wl-440)/(490-440); b = 1.0f; }
        else if (wl < 510) { g = 1.0f; b = -(wl-510)/(510-490); }
        else if (wl < 580) { r = (wl-510)/(580-510); g = 1.0f; }
        else if (wl < 645) { r = 1.0f; g = -(wl-645)/(645-580); }
        else if (wl <= 780) { r = 1.0f; }
        // Intensity falloff at edges
        float f = 1.0f;
        if (wl >= 380 && wl < 420) f = 0.3f + 0.7f*(wl-380)/(420-380);
        else if (wl >= 700) f = 0.3f + 0.7f*(780-wl)/(780-700);
        rgb[0]=r*f; rgb[1]=g*f; rgb[2]=b*f;
    }

    // ─── 15. Color Distance (Delta-E) ──────────────────────────────────────
    float color_distance(const float* lab1, const float* lab2, int metric) {
        float dL = lab1[0]-lab2[0], da = lab1[1]-lab2[1], db = lab1[2]-lab2[2];
        if (metric == 0) { // CIE76
            return std::sqrt(dL*dL + da*da + db*db);
        } else { // CIE94 simplified
            float C1 = std::sqrt(lab1[1]*lab1[1]+lab1[2]*lab1[2]);
            float C2 = std::sqrt(lab2[1]*lab2[1]+lab2[2]*lab2[2]);
            float dC = C1-C2;
            float dH2 = da*da + db*db - dC*dC;
            if (dH2 < 0) dH2 = 0;
            float SL=1, SC=1+0.045f*C1, SH=1+0.015f*C1;
            return std::sqrt((dL/SL)*(dL/SL) + (dC/SC)*(dC/SC) + dH2/(SH*SH));
        }
    }

    // ─── 16. Bradford Chromatic Adaptation ──────────────────────────────────
    void chromatic_adapt(const float* xyz_in, float* xyz_out, size_t num_pixels,
                         const float* src_w, const float* dst_w) {
        // Bradford matrix
        const float M[9] = {0.8951f,0.2664f,-0.1614f,-0.7502f,1.7135f,0.0367f,0.0389f,-0.0685f,1.0296f};
        float src_lms[3], dst_lms[3];
        for(int j=0;j<3;++j){src_lms[j]=0;dst_lms[j]=0;
            for(int k=0;k<3;++k){src_lms[j]+=M[j*3+k]*src_w[k];dst_lms[j]+=M[j*3+k]*dst_w[k];}}
        float scale[3]={dst_lms[0]/src_lms[0],dst_lms[1]/src_lms[1],dst_lms[2]/src_lms[2]};
        #pragma omp parallel for
        for(int i=0;i<(int)num_pixels;++i){
            float lms[3]={0,0,0};
            for(int j=0;j<3;++j) for(int k=0;k<3;++k) lms[j]+=M[j*3+k]*xyz_in[i*3+k];
            for(int j=0;j<3;++j) lms[j]*=scale[j];
            // Inverse Bradford (simplified)
            const float Mi[9]={0.9870f,-0.1471f,0.1600f,0.4323f,0.5184f,0.0493f,-0.0085f,0.0400f,0.9685f};
            for(int j=0;j<3;++j){xyz_out[i*3+j]=0;for(int k=0;k<3;++k)xyz_out[i*3+j]+=Mi[j*3+k]*lms[k];}
        }
    }

    // ─── 17. HDR Tonemapping ────────────────────────────────────────────────
    void hdr_tonemap(const float* hdr, float* ldr, size_t num_pixels, int method) {
        #pragma omp parallel for
        for (int i = 0; i < (int)(num_pixels*3); ++i) {
            float v = hdr[i];
            if (method == 0) // Reinhard
                ldr[i] = v / (1.0f + v);
            else if (method == 1) { // ACES
                float a=2.51f,b=0.03f,c=2.43f,d=0.59f,e=0.14f;
                ldr[i] = std::max(0.f, (v*(a*v+b))/(v*(c*v+d)+e));
            } else { // Filmic
                float x = std::max(0.f, v - 0.004f);
                ldr[i] = (x*(6.2f*x+0.5f))/(x*(6.2f*x+1.7f)+0.06f);
            }
        }
    }

    // ─── 18. Color Histogram ────────────────────────────────────────────────
    void color_histogram(const float* img, size_t num_pixels, int num_channels,
                         int num_bins, float* histogram_out) {
        std::memset(histogram_out, 0, num_channels * num_bins * sizeof(float));
        for (size_t i = 0; i < num_pixels; ++i) {
            for (int c = 0; c < num_channels; ++c) {
                float v = img[i * num_channels + c];
                int bin = std::min(num_bins - 1, std::max(0, (int)(v * num_bins)));
                histogram_out[c * num_bins + bin] += 1.0f;
            }
        }
    }

    // ─── 19. Dominant Colors (K-means) ──────────────────────────────────────
    void dominant_colors(const float* rgb, size_t num_pixels, int k,
                         float* palette_out, int max_iters) {
        // Initialize palette with evenly spaced samples
        for (int j = 0; j < k; ++j) {
            size_t idx = (j * num_pixels) / k;
            palette_out[j*3+0] = rgb[idx*3+0];
            palette_out[j*3+1] = rgb[idx*3+1];
            palette_out[j*3+2] = rgb[idx*3+2];
        }
        std::vector<int> assignments(num_pixels, 0);
        for (int iter = 0; iter < max_iters; ++iter) {
            // Assign
            #pragma omp parallel for
            for (int i = 0; i < (int)num_pixels; ++i) {
                float best_d = 1e9f; int best_k = 0;
                for (int j = 0; j < k; ++j) {
                    float dr=rgb[i*3]-palette_out[j*3], dg=rgb[i*3+1]-palette_out[j*3+1], db=rgb[i*3+2]-palette_out[j*3+2];
                    float d = dr*dr+dg*dg+db*db;
                    if (d < best_d) { best_d = d; best_k = j; }
                }
                assignments[i] = best_k;
            }
            // Update centroids
            std::vector<float> sums(k*3, 0); std::vector<int> counts(k, 0);
            for (size_t i = 0; i < num_pixels; ++i) {
                int j = assignments[i]; counts[j]++;
                sums[j*3+0]+=rgb[i*3]; sums[j*3+1]+=rgb[i*3+1]; sums[j*3+2]+=rgb[i*3+2];
            }
            for (int j = 0; j < k; ++j) {
                if (counts[j] > 0) {
                    palette_out[j*3+0]=sums[j*3+0]/counts[j];
                    palette_out[j*3+1]=sums[j*3+1]/counts[j];
                    palette_out[j*3+2]=sums[j*3+2]/counts[j];
                }
            }
        }
    }

    // ─── 20. Color Quantize ─────────────────────────────────────────────────
    void color_quantize(const float* rgb, float* out, size_t num_pixels,
                        const float* palette, int palette_size) {
        #pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i) {
            float best_d = 1e9f; int best = 0;
            for (int j = 0; j < palette_size; ++j) {
                float dr=rgb[i*3]-palette[j*3], dg=rgb[i*3+1]-palette[j*3+1], db=rgb[i*3+2]-palette[j*3+2];
                float d = dr*dr+dg*dg+db*db;
                if (d < best_d) { best_d = d; best = j; }
            }
            out[i*3+0]=palette[best*3]; out[i*3+1]=palette[best*3+1]; out[i*3+2]=palette[best*3+2];
        }
    }

} // namespace color
} // namespace vision
} // namespace mecan
