#include "mecan/distributed/collective.h"

#include <cstdlib>
#include <memory>
#include <string>

namespace mecan {
namespace distributed {

namespace {

class LocalCollectiveTransport final : public runtime::ICollectiveTransport {
public:
    std::string name() const override { return "Local"; }

    bool initialize(int world_size, int rank) override {
        world_size_ = world_size <= 0 ? 1 : world_size;
        rank_ = rank < 0 ? 0 : rank;
        return true;
    }

    void finalize() override {}

    int world_size() const override { return world_size_; }
    int rank() const override { return rank_; }

    bool allreduce_sum_f32(float*, size_t) override {
        // Single-process fallback. Real transports implement actual collectives.
        return true;
    }

    bool broadcast_f32(float*, size_t, int) override {
        return true;
    }

    bool barrier() override {
        return true;
    }

private:
    int world_size_ = 1;
    int rank_ = 0;
};

class PlaceholderCollectiveTransport final : public runtime::ICollectiveTransport {
public:
    explicit PlaceholderCollectiveTransport(std::string name)
        : name_(std::move(name)) {}

    std::string name() const override { return name_; }
    bool initialize(int world_size, int rank) override {
        fallback_.initialize(world_size, rank);
        return true;
    }
    void finalize() override { fallback_.finalize(); }
    int world_size() const override { return fallback_.world_size(); }
    int rank() const override { return fallback_.rank(); }
    bool allreduce_sum_f32(float* data, size_t count) override { return fallback_.allreduce_sum_f32(data, count); }
    bool broadcast_f32(float* data, size_t count, int root_rank) override { return fallback_.broadcast_f32(data, count, root_rank); }
    bool barrier() override { return fallback_.barrier(); }

private:
    std::string name_;
    LocalCollectiveTransport fallback_;
};

CollectiveBackend parse_env_backend() {
    const char* env = std::getenv("MECAN_COLLECTIVE");
    if (!env) return CollectiveBackend::Auto;

    const std::string v(env);
    if (v == "mpi") return CollectiveBackend::MPI;
    if (v == "nccl") return CollectiveBackend::NCCL;
    if (v == "rccl") return CollectiveBackend::RCCL;
    if (v == "gloo") return CollectiveBackend::Gloo;
    if (v == "local") return CollectiveBackend::Local;
    return CollectiveBackend::Auto;
}

} // namespace

std::shared_ptr<runtime::ICollectiveTransport> create_collective_transport(CollectiveBackend backend) {
    if (backend == CollectiveBackend::Auto) {
        backend = parse_env_backend();
    }
    if (backend == CollectiveBackend::Auto || backend == CollectiveBackend::Local) {
        return std::make_shared<LocalCollectiveTransport>();
    }
    if (backend == CollectiveBackend::MPI) {
        return std::make_shared<PlaceholderCollectiveTransport>("MPI");
    }
    if (backend == CollectiveBackend::NCCL) {
        return std::make_shared<PlaceholderCollectiveTransport>("NCCL");
    }
    if (backend == CollectiveBackend::RCCL) {
        return std::make_shared<PlaceholderCollectiveTransport>("RCCL");
    }
    return std::make_shared<PlaceholderCollectiveTransport>("Gloo");
}

} // namespace distributed
} // namespace mecan
