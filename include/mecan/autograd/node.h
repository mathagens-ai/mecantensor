// MecanTensor Autograd: Computation Graph Node
#pragma once

#include <vector>
#include <memory>

// Forward declaration of Tensor to avoid circular dependency
namespace mecan { class Tensor; }

namespace mecan {
namespace autograd {

// Forward declaration of Node
class Node;

// An Edge connects an output tensor to the Node that created it, and records 
// which input index it corresponds to in the backward pass.
struct Edge {
    std::shared_ptr<Node> function;
    uint32_t input_nr; // Index of the input to the target node
    
    Edge() : function(nullptr), input_nr(0) {}
    Edge(std::shared_ptr<Node> fn, uint32_t input_nr) 
        : function(std::move(fn)), input_nr(input_nr) {}
    
    bool is_valid() const { return function != nullptr; }
};

// Base class for all backward operations (the computation graph vertices)
class Node {
public:
    virtual ~Node() = default;

    // Execute the backward pass for this operation
    virtual std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) = 0;

    // Add an edge to a parent node (the nodes that created the inputs to this operation)
    void add_next_edge(Edge edge) {
        next_edges_.push_back(std::move(edge));
    }

    const std::vector<Edge>& next_edges() const {
        return next_edges_;
    }

private:
    std::vector<Edge> next_edges_;
};

} // namespace autograd
} // namespace mecan
