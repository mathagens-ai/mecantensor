#include "mecan/runtime/scheduler.h"

#include <utility>

namespace mecan {
namespace runtime {

AsyncToken AsyncScheduler::launch(std::string label, std::function<void()> task) {
    AsyncToken token;
    token.label = std::move(label);

    std::lock_guard<std::mutex> lock(mu_);
    token.id = next_id_++;
    tasks_[token.id] = std::async(std::launch::async, [task = std::move(task)]() {
        task();
    });
    return token;
}

void AsyncScheduler::wait(const AsyncToken& token) {
    std::future<void> fut;
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = tasks_.find(token.id);
        if (it == tasks_.end()) {
            return;
        }
        fut = std::move(it->second);
        tasks_.erase(it);
    }
    fut.wait();
}

void AsyncScheduler::wait_all() {
    std::vector<std::future<void>> futures;
    {
        std::lock_guard<std::mutex> lock(mu_);
        futures.reserve(tasks_.size());
        for (auto& kv : tasks_) {
            futures.push_back(std::move(kv.second));
        }
        tasks_.clear();
    }
    for (auto& f : futures) {
        f.wait();
    }
}

size_t AsyncScheduler::in_flight() const {
    std::lock_guard<std::mutex> lock(mu_);
    return tasks_.size();
}

} // namespace runtime
} // namespace mecan
