// MecanTensor Autograd Engine: Topological Sort & Graph Execution

#include "mecan/tensor.h"
#include "mecan/autograd/node.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <iostream>

namespace mecan {

    void Tensor::backward() {
        if (!this->requires_grad_) {
            throw std::runtime_error("backward() called on a tensor that does not require gradients.");
        }
        
        if (!this->grad_fn_) {
            // Leaf node backward (e.g. backward on a scalar parameter directly)
            if (!this->grad_ || !this->grad_->defined()) {
                // Initialize to 1.0 (assuming scalar output)
                // For non-scalars, backward() must receive a gradient argument in PyTorch
                // We'll stub it with 1.0 everywhere for the proof-of-concept
                Tensor ones(this->shape(), this->dtype(), this->device());
                // Fill with 1.0... (pseudo code)
                float* p = ones.data_ptr<float>();
                for (size_t i = 0; i < ones.numel(); ++i) p[i] = 1.0f;
                this->set_grad(ones);
            }
            return;
        }

        // Initialize grad to 1.0 if not set (for the root scalar)
        if (!this->grad_ || !this->grad_->defined()) {
            Tensor ones(this->shape(), this->dtype(), this->device());
            float* p = ones.data_ptr<float>();
            for (size_t i = 0; i < ones.numel(); ++i) p[i] = 1.0f;
            this->set_grad(ones);
        }

        // 1. Topological Sort: Compute in-degrees of nodes to evaluate in reverse topological order
        std::unordered_map<autograd::Node*, int> in_degrees;
        std::unordered_set<autograd::Node*> visited;
        std::queue<autograd::Node*> q;
        
        q.push(this->grad_fn_.get());
        
        while (!q.empty()) {
            auto node = q.front();
            q.pop();
            
            if (visited.find(node) == visited.end()) {
                visited.insert(node);
                for (const auto& edge : node->next_edges()) {
                    if (edge.is_valid()) {
                        in_degrees[edge.function.get()]++;
                        q.push(edge.function.get());
                    }
                }
            }
        }

        // 2. Execute backwards using Kahn's algorithm
        // Queue holds nodes with in-degree == 0 (ready to compute)
        std::queue<autograd::Node*> exec_queue;
        exec_queue.push(this->grad_fn_.get());

        // Map to store accumulated gradients for each node
        // In PyTorch, nodes receive a vector of Tensor (one grad per output edge)
        std::unordered_map<autograd::Node*, std::vector<Tensor>> node_grads;
        
        // Root gets its own grad
        node_grads[this->grad_fn_.get()].push_back(this->grad());

        while (!exec_queue.empty()) {
            auto node = exec_queue.front();
            exec_queue.pop();

            // Fetch accumulated gradients for this node
            const auto& grad_outputs = node_grads[node];

            // Apply backward function!
            std::vector<Tensor> grad_inputs = node->apply(grad_outputs);

            // Distribute gradients to parents
            const auto& next_edges = node->next_edges();
            for (size_t i = 0; i < next_edges.size(); ++i) {
                const auto& edge = next_edges[i];
                if (edge.is_valid()) {
                    auto next_node = edge.function.get();
                    
                    // Accumulate gradient for the next node's specific input
                    if (node_grads[next_node].size() <= edge.input_nr) {
                        node_grads[next_node].resize(edge.input_nr + 1);
                    }
                    
                    if (!node_grads[next_node][edge.input_nr].defined()) {
                        node_grads[next_node][edge.input_nr] = grad_inputs[i];
                    } else {
                        // TODO: Implement actual tensor addition for accumulating gradients from multiple branches
                        // ops::add(node_grads[next_node][edge.input_nr], grad_inputs[i], node_grads[next_node][edge.input_nr]);
                    }

                    // Decrement in-degree, if 0, push to exec_queue
                    in_degrees[next_node]--;
                    if (in_degrees[next_node] == 0) {
                        exec_queue.push(next_node);
                    }
                }
            }
        }
    }

} // namespace mecan
