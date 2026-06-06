#pragma once
#include "mecan/tensor.h"
#include "mecan/optim/lgc.h"
#include "mecan/nn/module.h"
#include "mecan/nn/loss.h"
#include <memory>

namespace mecan {

    /**
     * Native C++ Trainer (Track 4)
     * ----------------------------
     * Bypasses the Python Global Interpreter Lock (GIL).
     * Runs the entire Forward -> Loss -> Backward -> Step 
     * loop entirely in C++ at maximum hardware speed.
     */
    class Trainer {
    private:
        std::shared_ptr<nn::Module> model_;
        std::shared_ptr<optim::LGCOptimizer> optimizer_;

    public:
        Trainer(std::shared_ptr<nn::Module> model, float lr);

        // Executes one full training step and returns the MAE Loss
        float train_step(const Tensor& input, const Tensor& target);
    };

} // namespace mecan
