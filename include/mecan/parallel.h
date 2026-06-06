#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <omp.h>
#include <algorithm>
#include <utility>
#include <vector>

namespace mecan {
namespace parallel {

    /**
     * TST Thread Management
     * creating race conditions or OS starvation.
     */

    inline int get_num_threads() {
        return omp_get_max_threads();
    }

    inline void set_num_threads(int n) {
        omp_set_num_threads(std::max(1, n));
    }

    // A helper to execute a loop in parallel with a specified grain size
    template <typename F>
    void parallel_for(int64_t begin, int64_t end, int64_t grain_size, F f) {
        if (begin >= end) return;
        
        if (omp_in_parallel() || (end - begin) < grain_size) {
            for (int64_t i = begin; i < end; ++i) f(i);
        } else {
            #pragma omp parallel for
            for (int64_t i = begin; i < end; ++i) f(i);
        }
    }

    struct LaunchGraphStats {
        double sequential_ms = 0.0;
        double graph_ms = 0.0;
        double estimated_time_saved_ms = 0.0;
    };

    class LaunchGraph {
    public:
        void add_kernel(std::function<void()> kernel) {
            kernels_.push_back(std::move(kernel));
        }

        void clear() {
            kernels_.clear();
        }

        void launch() const {
            for (const auto& k : kernels_) {
                k();
            }
        }

        LaunchGraphStats benchmark(int repeats = 10, double launch_overhead_ms = 0.02) const {
            LaunchGraphStats stats{};
            if (kernels_.empty() || repeats <= 0) {
                return stats;
            }

            // Baseline: sequential launches pay overhead per kernel.
            const auto t0 = std::chrono::high_resolution_clock::now();
            for (int r = 0; r < repeats; ++r) {
                for (const auto& k : kernels_) {
                    k();
                }
            }
            const auto t1 = std::chrono::high_resolution_clock::now();
            stats.sequential_ms = std::chrono::duration<double, std::milli>(t1 - t0).count()
                                  + (launch_overhead_ms * kernels_.size() * repeats);

            // Graph launch: overhead paid once per launch.
            const auto t2 = std::chrono::high_resolution_clock::now();
            for (int r = 0; r < repeats; ++r) {
                launch();
            }
            const auto t3 = std::chrono::high_resolution_clock::now();
            stats.graph_ms = std::chrono::duration<double, std::milli>(t3 - t2).count()
                             + (launch_overhead_ms * repeats);

            stats.estimated_time_saved_ms = std::max(0.0, stats.sequential_ms - stats.graph_ms);
            return stats;
        }

    private:
        std::vector<std::function<void()>> kernels_;
    };

} // namespace parallel
} // namespace mecan
