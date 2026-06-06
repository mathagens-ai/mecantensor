// MecanTensor: TSI (Temporal Synaptic Infusion)
// Infuses sequence context into a fixed-size resonant state vector,
// achieving O(1) memory scaling regardless of context length.
#pragma once

#include "../tensor.h"

namespace mecan {
namespace ops {

    // Infuses the current token's state into the fixed-size Resonance State (R)
    // R_t = R_{t-1} * decay + (Query @ Key) * Value (Stateful folding)
    void tsi_infuse(
        const Tensor& current_token, // [Batch, Hidden]
        Tensor& resonant_state,      // [Batch, Hidden, Hidden] - The folded context
        const Tensor& infusion_gate  // [Batch, Hidden] - Learned infusion rate from TSSR
    );

    // Reads from the folded Resonance State to generate the output token
    void tsi_read(
        const Tensor& current_token, // [Batch, Hidden]
        const Tensor& resonant_state,// [Batch, Hidden, Hidden]
        Tensor& output               // [Batch, Hidden]
    );

} // namespace ops
} // namespace mecan
