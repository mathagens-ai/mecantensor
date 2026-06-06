#pragma once

#include <cstdint>
#include <functional>
#include <future>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mecan {
namespace runtime {

struct AsyncToken {
    uint64_t id = 0;
    std::string label;
};

// Unified asynchronous task scheduler with stream/event-like semantics.
class AsyncScheduler {
public:
    AsyncToken launch(std::string label, std::function<void()> task);
    void wait(const AsyncToken& token);
    void wait_all();
    size_t in_flight() const;

private:
    mutable std::mutex mu_;
    uint64_t next_id_ = 1;
    std::unordered_map<uint64_t, std::future<void>> tasks_;
};

} // namespace runtime
} // namespace mecan
