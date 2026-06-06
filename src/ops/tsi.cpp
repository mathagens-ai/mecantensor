// MecanTensor: TSI (Temporal Synaptic Infusion) Implementation

#include "mecan/ops/tsi.h"
#include "mecan/autograd/functions.h"
#include "../hlas/hlas.h"
#include <stdexcept>
#include <omp.h>

namespace mecan {
namespace ops {

    void tsi_infuse(const Tensor& current_token, Tensor& resonant_state, const Tensor& infusion_gate) {
        if (current_token.ndimension() != 2 || resonant_state.ndimension() != 3 || infusion_gate.ndimension() != 2) {
            throw std::runtime_error("TSI Error: Invalid dimensions. Expected Token[B, H], State[B, H, H], Gate[B, H]");
        }

        const size_t batch = current_token.size(0);
        const size_t hidden = current_token.size(1);

        if (resonant_state.size(0) != batch || resonant_state.size(1) != hidden || resonant_state.size(2) != hidden) {
            throw std::runtime_error("TSI Error: Resonant State must be [Batch, Hidden, Hidden]");
        }

        const float* token_ptr = current_token.data_ptr<float>();
        const float* gate_ptr = infusion_gate.data_ptr<float>();
        float* state_ptr = resonant_state.data_ptr<float>();

        // We use OpenMP here because this is a batch of Rank-1 updates.
        // For massive hidden sizes, this is automatically routed to the HLAS engine.
        #pragma omp parallel for schedule(static)
        for (int64_t b = 0; b < static_cast<int64_t>(batch); ++b) {
            const float* t_b = token_ptr + b * hidden;
            const float* g_b = gate_ptr + b * hidden;
            float* s_b = state_ptr + b * hidden * hidden;

            // R_t = (R_{t-1} * decay) + (Token_K ⊗ Token_V)
            // We use the infusion gate as a dynamic decay/infusion mechanism derived from TSSR.
            for (size_t i = 0; i < hidden; ++i) {
                float k_val = t_b[i];
                float decay = g_b[i]; // Learned resonance frequency
                
                for (size_t j = 0; j < hidden; ++j) {
                    float v_val = t_b[j];
                    // The core QCF fold: State is decayed by the gate, and infused with the outer product
                    s_b[i * hidden + j] = (s_b[i * hidden + j] * decay) + (k_val * v_val);
                }
            }
        }

        // ─── Autograd Forward Linkage (Infusion + LGC) ──────────────────────
        if (current_token.requires_grad() || resonant_state.requires_grad() || infusion_gate.requires_grad()) {
            resonant_state.set_requires_grad(true);
            auto backward_fn = std::make_shared<autograd::TsiInfuseBackward>(current_token, resonant_state, infusion_gate);
            
            auto add_edge = [&](const Tensor& t) {
                if (t.requires_grad()) {
                    if (t.grad_fn()) backward_fn->add_next_edge(autograd::Edge(t.grad_fn(), 0));
                    else backward_fn->add_next_edge(autograd::Edge(std::make_shared<autograd::AccumulateGrad>(std::make_shared<Tensor>(t)), 0));
                } else {
                    backward_fn->add_next_edge(autograd::Edge());
                }
            };

            add_edge(current_token);
            add_edge(resonant_state);
            add_edge(infusion_gate);

            resonant_state.set_grad_fn(backward_fn);
        }
    }

    // ─── QCF: Resonant State Read ──────────────────────────────────────────
    void tsi_read(const Tensor& current_token, const Tensor& resonant_state, Tensor& output) {
        const size_t batch = current_token.size(0);
        const size_t hidden = current_token.size(1);

        const float* token_ptr = current_token.data_ptr<float>();
        const float* state_ptr = resonant_state.data_ptr<float>();
        float* out_ptr = output.data_ptr<float>();

        // Y_t = R_t @ Q_t
        // We execute a batch of Matrix-Vector multiplications.
        for (int64_t b = 0; b < static_cast<int64_t>(batch); ++b) {
            const float* s_b = state_ptr + b * hidden * hidden;
            const float* t_b = token_ptr + b * hidden;
            float* o_b = out_ptr + b * hidden;

            // Route Matrix-Vector multiplication through the HLAS engine
            // Matrix: [Hidden x Hidden], Vector: [Hidden x 1] -> Output: [Hidden x 1]
            mecan::hlas::get_engine()->sgemm(
                static_cast<int>(hidden), 
                1, 
                static_cast<int>(hidden), 
                1.0f, 
                s_b, static_cast<int>(hidden), 
                t_b, 1, 
                0.0f, 
                o_b, 1, 
                mecan::hlas::FusedActivation::NONE
            );
        }

        // ─── Autograd Forward Linkage ──────────────────────────────────────────
        if (current_token.requires_grad() || resonant_state.requires_grad()) {
            output.set_requires_grad(true);
            auto backward_fn = std::make_shared<autograd::TsiReadBackward>(current_token, resonant_state);
            
            auto add_edge = [&](const Tensor& t) {
                if (t.requires_grad()) {
                    if (t.grad_fn()) backward_fn->add_next_edge(autograd::Edge(t.grad_fn(), 0));
                    else backward_fn->add_next_edge(autograd::Edge(std::make_shared<autograd::AccumulateGrad>(std::make_shared<Tensor>(t)), 0));
                } else {
                    backward_fn->add_next_edge(autograd::Edge());
                }
            };

            add_edge(current_token);
            add_edge(resonant_state);

            output.set_grad_fn(backward_fn);
        }
    }

} // namespace ops
} // namespace mecan
