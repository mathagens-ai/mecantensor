#include "mecan/trainer.h"
#include "mecan/autograd/engine.h"
#include <vector>

namespace mecan {

    Trainer::Trainer(std::shared_ptr<nn::Module> model, float lr) 
        : model_(model) {
        
        // Initialize the AdamW-killer LGC Optimizer
        // Assumes model exposes a parameter count or list
        // For the 100B Core, we use the fused LGC config
        optimizer_ = std::make_shared<optim::LGCOptimizer>(1000000, lr, 0.9f);
    }

    float Trainer::train_step(const Tensor& input, const Tensor& target) {
        // 1. Forward Pass (Hardware Agnostic, parallelized via OpenMP)
        autograd::Variable in_var(input, false);
        autograd::Variable pred = model_->forward(in_var);

        // 2. Fused Loss & Gradient Injection (Track 4)
        // Avoids allocating the massive N-dimensional Error Tensor
        float loss_val = nn::FusedLossMAE::compute_and_backward(pred, target);

        // 3. Autograd Engine Backward Graph traversal (Track 1)
        autograd::Engine::backward(pred);

        // 4. LGC Optimizer Step (SSD Infusion Aware)
        // Directly maps gradients to the SSD-backed weights
        std::vector<autograd::Variable*> params = model_->parameters();
        optimizer_->step(params);

        // 5. Zero Gradients
        optimizer_->zero_grad(params);

        return loss_val;
    }

} // namespace mecan
