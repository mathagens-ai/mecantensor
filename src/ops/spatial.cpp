#include "mecan/ops/spatial.h"
#include <omp.h>

namespace mecan {
namespace ops {

    void zero_pad_strided_upscale(const Tensor& input, Tensor& output, int stride) {
        // Instead of allocating a massive zero-padded matrix in RAM,
        // we directly map the input grid onto the larger output grid
        // using pointer math. The output is assumed pre-allocated with zeros.
        // This defeats the memory bottleneck of PyTorch ConvTranspose2d.
        
        size_t out_h = output.size(2);
        size_t out_w = output.size(3);
        size_t in_h = input.size(2);
        size_t in_w = input.size(3);

        const float* in_ptr = input.data_ptr<float>();
        float* out_ptr = output.data_ptr<float>();

        #pragma omp parallel for collapse(2)
        for (int64_t y = 0; y < (int64_t)in_h; ++y) {
            for (int64_t x = 0; x < (int64_t)in_w; ++x) {
                size_t out_y = y * stride;
                size_t out_x = x * stride;
                
                // Direct mapping, bypassing padding allocation
                out_ptr[out_y * out_w + out_x] = in_ptr[y * in_w + x];
            }
        }
    }

} // namespace ops
} // namespace mecan
