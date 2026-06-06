#include "autograd/ops.h"

#include <stdexcept>
#include <utility>
#include <vector>

#include "autograd/function.h"
#include "autograd/trace.h"
#include "ops/math.h"

namespace mecan {
namespace autograd {
namespace {

void add_inplace_tensor(Tensor& dst, const Tensor& src) {
    if (dst.numel() != src.numel()) {
        throw std::runtime_error("Autograd add_inplace mismatch.");
    }
    if (dst.dtype() != src.dtype()) {
        throw std::runtime_error("Autograd add_inplace dtype mismatch.");
    }
    if (dst.dtype() != core::ScalarType::Float32) {
        throw std::runtime_error("Autograd add_inplace currently supports Float32 tensors only.");
    }

    float* dst_ptr = dst.data_ptr<float>();
    const float* src_ptr = src.data_ptr<float>();
    for (size_t i = 0; i < dst.numel(); ++i) {
        dst_ptr[i] += src_ptr[i];
    }
}

class AccumulateGrad : public Function {
public:
    explicit AccumulateGrad(Variable* target) : Function("AccumulateGrad"), target_(target) {}

    std::vector<Variable> apply(const std::vector<Variable>& grads) override {
        if (!target_ || grads.empty() || !target_->requires_grad) {
            return {};
        }

        if (target_->grad.numel() == 0) {
            target_->grad = Tensor(target_->data.shape(), target_->data.dtype(), target_->data.device());
            target_->zero_grad();
        }
        add_inplace_tensor(target_->grad, grads[0].data);
        return {};
    }

private:
    Variable* target_;
};

std::shared_ptr<Function> ensure_accumulator(Variable& v) {
    if (!v.requires_grad) {
        return nullptr;
    }
    if (!v.grad_accumulator) {
        v.grad_accumulator = std::make_shared<AccumulateGrad>(&v);
    }
    return v.grad_accumulator;
}

std::shared_ptr<Function> next_edge_for(Variable& v) {
    if (!v.requires_grad) {
        return nullptr;
    }
    if (v.grad_fn) {
        return v.grad_fn;
    }
    return ensure_accumulator(v);
}

class AddBackward : public Function {
public:
    AddBackward(bool need_a, bool need_b)
        : Function("AddBackward"), need_a_(need_a), need_b_(need_b) {}

    std::vector<Variable> apply(const std::vector<Variable>& grads) override {
        if (grads.empty()) {
            return {};
        }
        std::vector<Variable> out;
        if (need_a_) out.push_back(grads[0]);
        if (need_b_) out.push_back(grads[0]);
        return out;
    }

private:
    bool need_a_;
    bool need_b_;
};

class ReluBackward : public Function {
public:
    explicit ReluBackward(Tensor input) : Function("ReluBackward"), input_(std::move(input)) {}

    std::vector<Variable> apply(const std::vector<Variable>& grads) override {
        if (grads.empty()) {
            return {};
        }
        if (input_.dtype() != core::ScalarType::Float32 || grads[0].data.dtype() != core::ScalarType::Float32) {
            throw std::runtime_error("ReLU backward currently supports Float32 tensors only.");
        }

        Tensor grad_in(input_.shape(), input_.dtype(), input_.device());
        const float* inp = input_.data_ptr<float>();
        const float* gout = grads[0].data.data_ptr<float>();
        float* gin = grad_in.data_ptr<float>();

        for (size_t i = 0; i < input_.numel(); ++i) {
            gin[i] = (inp[i] > 0.0f) ? gout[i] : 0.0f;
        }
        return {Variable(grad_in, false)};
    }

private:
    Tensor input_;
};

class MatmulBackward : public Function {
public:
    MatmulBackward(Tensor a, Tensor b, bool need_a, bool need_b)
        : Function("MatmulBackward"), a_(std::move(a)), b_(std::move(b)), need_a_(need_a), need_b_(need_b) {}

    std::vector<Variable> apply(const std::vector<Variable>& grads) override {
        if (grads.empty()) {
            return {};
        }
        if (a_.dtype() != core::ScalarType::Float32 ||
            b_.dtype() != core::ScalarType::Float32 ||
            grads[0].data.dtype() != core::ScalarType::Float32) {
            throw std::runtime_error("Matmul backward currently supports Float32 tensors only.");
        }
        if (a_.ndimension() != 2 || b_.ndimension() != 2 || grads[0].data.ndimension() != 2) {
            throw std::runtime_error("Matmul backward expects 2D tensors.");
        }

        const size_t M = a_.size(0);
        const size_t K = a_.size(1);
        const size_t N = b_.size(1);

        Tensor grad_a({M, K}, core::ScalarType::Float32, a_.device());
        Tensor grad_b({K, N}, core::ScalarType::Float32, b_.device());
        Tensor grad_out = grads[0].data;

        const float* a_ptr = a_.data_ptr<float>();
        const float* b_ptr = b_.data_ptr<float>();
        const float* g_ptr = grad_out.data_ptr<float>();
        float* ga_ptr = grad_a.data_ptr<float>();
        float* gb_ptr = grad_b.data_ptr<float>();

        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                float sum = 0.0f;
                for (size_t j = 0; j < N; ++j) {
                    sum += g_ptr[i * N + j] * b_ptr[k * N + j];
                }
                ga_ptr[i * K + k] = sum;
            }
        }

        for (size_t k = 0; k < K; ++k) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t i = 0; i < M; ++i) {
                    sum += a_ptr[i * K + k] * g_ptr[i * N + j];
                }
                gb_ptr[k * N + j] = sum;
            }
        }

        std::vector<Variable> out;
        if (need_a_) out.emplace_back(grad_a, false);
        if (need_b_) out.emplace_back(grad_b, false);
        return out;
    }

private:
    Tensor a_;
    Tensor b_;
    bool need_a_;
    bool need_b_;
};

} // namespace

namespace functional {

Variable add(Variable& a, Variable& b) {
    Tensor out_data(a.data.shape(), a.data.dtype(), a.data.device());
    ops::add(a.data, b.data, out_data);

    Variable out(out_data, a.requires_grad || b.requires_grad);
    auto trace = TraceTape::pre_call("add", {&a, &b});

    if (out.requires_grad) {
        auto backward_fn = std::make_shared<AddBackward>(a.requires_grad, b.requires_grad);
        if (a.requires_grad) backward_fn->add_next_function(next_edge_for(a));
        if (b.requires_grad) backward_fn->add_next_function(next_edge_for(b));
        out.grad_fn = backward_fn;
    }

    TraceTape::post_call(trace, out);
    return out;
}

Variable matmul(Variable& a, Variable& b) {
    if (a.data.ndimension() != 2 || b.data.ndimension() != 2) {
        throw std::runtime_error("Autograd matmul expects 2D tensors.");
    }
    Tensor out_data({a.data.size(0), b.data.size(1)}, a.data.dtype(), a.data.device());
    ops::matmul(a.data, b.data, out_data);

    Variable out(out_data, a.requires_grad || b.requires_grad);
    auto trace = TraceTape::pre_call("matmul", {&a, &b});

    if (out.requires_grad) {
        auto backward_fn = std::make_shared<MatmulBackward>(a.data, b.data, a.requires_grad, b.requires_grad);
        if (a.requires_grad) backward_fn->add_next_function(next_edge_for(a));
        if (b.requires_grad) backward_fn->add_next_function(next_edge_for(b));
        out.grad_fn = backward_fn;
    }

    TraceTape::post_call(trace, out);
    return out;
}

Variable relu(Variable& x) {
    Tensor out_data(x.data.shape(), x.data.dtype(), x.data.device());
    if (x.data.dtype() != core::ScalarType::Float32) {
        throw std::runtime_error("Autograd relu currently supports Float32 tensors only.");
    }

    const float* in_ptr = x.data.data_ptr<float>();
    float* out_ptr = out_data.data_ptr<float>();
    for (size_t i = 0; i < x.data.numel(); ++i) {
        out_ptr[i] = (in_ptr[i] > 0.0f) ? in_ptr[i] : 0.0f;
    }

    Variable out(out_data, x.requires_grad);
    auto trace = TraceTape::pre_call("relu", {&x});

    if (out.requires_grad) {
        auto backward_fn = std::make_shared<ReluBackward>(x.data);
        backward_fn->add_next_function(next_edge_for(x));
        out.grad_fn = backward_fn;
    }

    TraceTape::post_call(trace, out);
    return out;
}

} // namespace functional
} // namespace autograd
} // namespace mecan
