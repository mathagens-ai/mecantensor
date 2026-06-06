#include "mecan/ops/pad.h"
#include <omp.h>
#include <algorithm>

namespace mecan {
namespace ops {

void pad2d(const float* input, float* output,
           size_t N, size_t C, size_t H, size_t W,
           int pad_left, int pad_right, int pad_top, int pad_bottom,
           int mode, float constant_value) {
           
    size_t out_H = H + pad_top + pad_bottom;
    size_t out_W = W + pad_left + pad_right;
    
    #pragma omp parallel for collapse(4)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < out_H; ++oh) {
                for (int ow = 0; ow < out_W; ++ow) {
                    
                    int ih = oh - pad_top;
                    int iw = ow - pad_left;
                    
                    float val = constant_value;
                    
                    if (mode == 0) { // Constant
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            val = input[n * C * H * W + c * H * W + ih * W + iw];
                        }
                    } else if (mode == 1) { // Reflect
                        if (ih < 0) ih = -ih;
                        else if (ih >= H) ih = H - 1 - (ih - H + 1);
                        
                        if (iw < 0) iw = -iw;
                        else if (iw >= W) iw = W - 1 - (iw - W + 1);
                        
                        // clamp just in case
                        ih = std::max(0, std::min((int)H - 1, ih));
                        iw = std::max(0, std::min((int)W - 1, iw));
                        
                        val = input[n * C * H * W + c * H * W + ih * W + iw];
                    } else if (mode == 2) { // Replicate
                        ih = std::max(0, std::min((int)H - 1, ih));
                        iw = std::max(0, std::min((int)W - 1, iw));
                        val = input[n * C * H * W + c * H * W + ih * W + iw];
                    }
                    
                    output[n * C * out_H * out_W + c * out_H * out_W + oh * out_W + ow] = val;
                }
            }
        }
    }
}

} // namespace ops
} // namespace mecan
