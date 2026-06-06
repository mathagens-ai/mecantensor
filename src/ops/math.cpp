// MecanTensor Core Math — OpenBLAS MatMul + AVX2 Element-Wise Operations
#include "ops/math.h"
#include "mecan/autograd/functions.h"
#include "../hlas/hlas.h"
#include <omp.h>
#include <stdexcept>
#include <immintrin.h>
#if defined(__has_include)
#if __has_include(<cblas.h>)
#include <cblas.h>
#define MECAN_HAS_CBLAS 1
#endif
#endif

#if defined(_MSC_VER)
    #include <intrin.h>
#endif

namespace mecan {
namespace ops {

    // ─── AVX2 Vectorized Element-wise Addition ──────────────────────────────
    void add(const Tensor& a, const Tensor& b, Tensor& out) {
        if (a.numel() != b.numel() || a.numel() != out.numel()) {
            throw std::runtime_error("TST Error: Tensor shape mismatch in 'add'.");
        }

        if (a.dtype() == core::ScalarType::Float32) {
            const float* a_ptr = a.data_ptr<float>();
            const float* b_ptr = b.data_ptr<float>();
            float* out_ptr = out.data_ptr<float>();
            int64_t n = (int64_t)a.numel();

            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < n; i += 8) {
#if defined(__AVX2__)
                if (i + 8 <= n) {
                    __m256 va = _mm256_loadu_ps(a_ptr + i);
                    __m256 vb = _mm256_loadu_ps(b_ptr + i);
                    _mm256_storeu_ps(out_ptr + i, _mm256_add_ps(va, vb));
                } else {
                    for (int64_t j = i; j < n; ++j)
                        out_ptr[j] = a_ptr[j] + b_ptr[j];
                }
#else
                int64_t end = (i + 8 < n) ? i + 8 : n;
                for (int64_t j = i; j < end; ++j)
                    out_ptr[j] = a_ptr[j] + b_ptr[j];
#endif
            }
        } else {
            throw std::runtime_error("TST Error: DType not yet implemented for 'add'.");
        }
    }

    void sub(const Tensor& a, const Tensor& b, Tensor& out) {
        if (a.numel() != b.numel() || a.numel() != out.numel()) {
            throw std::runtime_error("TST Error: Tensor shape mismatch in 'sub'.");
        }
        if (a.dtype() != core::ScalarType::Float32 || b.dtype() != core::ScalarType::Float32 ||
            out.dtype() != core::ScalarType::Float32) {
            throw std::runtime_error("TST Error: 'sub' currently supports Float32 only.");
        }

        const float* a_ptr = a.data_ptr<float>();
        const float* b_ptr = b.data_ptr<float>();
        float* out_ptr = out.data_ptr<float>();
        const int64_t n = static_cast<int64_t>(a.numel());

        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            out_ptr[i] = a_ptr[i] - b_ptr[i];
        }
    }

    void mul(const Tensor& a, const Tensor& b, Tensor& out) {
        if (a.numel() != b.numel() || a.numel() != out.numel()) {
            throw std::runtime_error("TST Error: Tensor shape mismatch in 'mul'.");
        }
        if (a.dtype() != core::ScalarType::Float32 || b.dtype() != core::ScalarType::Float32 ||
            out.dtype() != core::ScalarType::Float32) {
            throw std::runtime_error("TST Error: 'mul' currently supports Float32 only.");
        }

        const float* a_ptr = a.data_ptr<float>();
        const float* b_ptr = b.data_ptr<float>();
        float* out_ptr = out.data_ptr<float>();
        const int64_t n = static_cast<int64_t>(a.numel());

        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            out_ptr[i] = a_ptr[i] * b_ptr[i];
        }
    }

// ─── HLAS Universal MatMul ────────────────────────────────────────────────
void matmul(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.ndimension() != 2 || b.ndimension() != 2) {
        throw std::runtime_error("TST Error: Matmul currently only supports 2D tensors.");
    }

    size_t M = a.size(0);
    size_t K = a.size(1);
    size_t N = b.size(1);

    if (b.size(0) != K || out.size(0) != M || out.size(1) != N) {
        throw std::runtime_error("TST Error: Incompatible shapes for matmul.");
    }

    const float* A = a.data_ptr<float>();
    const float* B = b.data_ptr<float>();
    float* C = out.data_ptr<float>();

    // Route through the Universal HLAS Router!
    mecan::hlas::get_engine()->sgemm(
        static_cast<int>(M), 
        static_cast<int>(N), 
        static_cast<int>(K), 
        1.0f, A, static_cast<int>(K), 
        B, static_cast<int>(N), 
        0.0f, C, static_cast<int>(N)
    );

    // ─── Autograd Forward Linkage ──────────────────────────────────────────
    if (a.requires_grad() || b.requires_grad()) {
        out.set_requires_grad(true);
        auto backward_fn = std::make_shared<autograd::MatMulBackward>(a, b);
        
        if (a.requires_grad()) {
            if (a.grad_fn()) backward_fn->add_next_edge(autograd::Edge(a.grad_fn(), 0));
            else backward_fn->add_next_edge(autograd::Edge(std::make_shared<autograd::AccumulateGrad>(std::make_shared<Tensor>(a)), 0));
        } else {
            backward_fn->add_next_edge(autograd::Edge()); // Null edge
        }

        if (b.requires_grad()) {
            if (b.grad_fn()) backward_fn->add_next_edge(autograd::Edge(b.grad_fn(), 0));
            else backward_fn->add_next_edge(autograd::Edge(std::make_shared<autograd::AccumulateGrad>(std::make_shared<Tensor>(b)), 0));
        } else {
            backward_fn->add_next_edge(autograd::Edge());
        }

        out.set_grad_fn(backward_fn);
    }
}

    // ─── Ternary Threshold ──────────────────────────────────────────────────
    void ternary_threshold(const Tensor& input, Tensor& output, float threshold) {
        const float* in = input.data_ptr<float>();
        int8_t* out = output.data_ptr<int8_t>();
        size_t n = input.numel();

        #pragma omp parallel for
        for (int64_t i = 0; i < (int64_t)n; ++i) {
            if (in[i] > threshold) out[i] = 1;
            else if (in[i] < -threshold) out[i] = -1;
            else out[i] = 0;
        }
    }

} // namespace ops
} // namespace mecan
