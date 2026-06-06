#include "mecan/nn/ntm.h"
#include <omp.h>

namespace mecan {
namespace nn {

    TernaryMemoryRouter::TernaryMemoryRouter(size_t slots, size_t dim) {
        // Pre-allocate memory bank on SSD
        memory_bank_ = Tensor({slots, dim}, core::ScalarType::Float32, core::Device(core::DeviceType::SSD_Infusion));
    }

    void TernaryMemoryRouter::read_head(const Tensor& ternary_address, Tensor& output) {
        // Ternary address is a direct index mask, no softmax needed.
        const int8_t* addr = ternary_address.data_ptr<int8_t>();
        const float* mem = memory_bank_.data_ptr<float>();
        float* out = output.data_ptr<float>();

        size_t slots = memory_bank_.size(0);
        size_t dim = memory_bank_.size(1);

        #pragma omp parallel for
        for (int64_t d = 0; d < (int64_t)dim; ++d) {
            float sum = 0.0f;
            for (int64_t s = 0; s < (int64_t)slots; ++s) {
                if (addr[s] == 1) sum += mem[s * dim + d];
                else if (addr[s] == -1) sum -= mem[s * dim + d];
            }
            out[d] = sum;
        }
    }

    void TernaryMemoryRouter::write_head(const Tensor& ternary_address, const Tensor& data) {
         // Omitted for brevity, symmetrical to read_head using ternary masking to write
    }

} // namespace nn
} // namespace mecan
