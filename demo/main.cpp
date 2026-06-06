#include <iostream>
#include <vector>
#include <fstream>
#include "mecan/mecan.h"

int main() {
    mecan::info();

    // 1. Create Tensors
    std::vector<size_t> shape = {1024, 1024};
    mecan::Tensor a(shape, mecan::core::ScalarType::Float32);
    mecan::Tensor b(shape, mecan::core::ScalarType::Float32);
    mecan::Tensor c(shape, mecan::core::ScalarType::Float32);

    // Initialize data
    float* a_raw = a.data_ptr<float>();
    for (size_t i = 0; i < a.numel(); ++i) a_raw[i] = 1.0f;

    float* b_raw = b.data_ptr<float>();
    for (size_t i = 0; i < b.numel(); ++i) b_raw[i] = 2.0f;

    std::cout << "Tensors initialized. Executing Matmul (1024x1024)..." << std::endl;

    // 2. Execute Math
    mecan::ops::matmul(a, b, c);

    std::cout << "Matmul Complete. Top left value: " << c.data_ptr<float>()[0] << " (Expected 2048)" << std::endl;

    // 3. Save/Load Test
    std::cout << "\n=== I/O Serialization Demo ===" << std::endl;
    mecan::io::save(c, "weights.mt");
    mecan::Tensor c_loaded = mecan::io::load("weights.mt");
    std::cout << "Loaded tensor 'c' from 'weights.mt'." << std::endl;
    std::cout << "Load Complete. Data validity check: " << c_loaded.data_ptr<float>()[0] << std::endl;

    // 4. LaunchGraph demonstration (CPU launch overhead amortization)
    std::vector<float> launch_buf(1 << 18, 1.0f);
    mecan::parallel::LaunchGraph launch_graph;
    launch_graph.add_kernel([&]() {
        for (size_t i = 0; i < launch_buf.size(); i += 4) launch_buf[i] += 0.1f;
    });
    launch_graph.add_kernel([&]() {
        for (size_t i = 1; i < launch_buf.size(); i += 4) launch_buf[i] += 0.2f;
    });
    launch_graph.add_kernel([&]() {
        for (size_t i = 2; i < launch_buf.size(); i += 4) launch_buf[i] += 0.3f;
    });
    launch_graph.add_kernel([&]() {
        for (size_t i = 3; i < launch_buf.size(); i += 4) launch_buf[i] += 0.4f;
    });

    const auto launch_stats = launch_graph.benchmark(20, 0.03);
    std::cout << "LaunchGraph benchmark -> sequential_ms=" << launch_stats.sequential_ms
              << " graph_ms=" << launch_stats.graph_ms
              << " saved_ms=" << launch_stats.estimated_time_saved_ms << std::endl;

    // 5. Autograd + graph tracing demonstration (Linear -> ReLU -> Linear)
    using mecan::autograd::Variable;
    using mecan::autograd::TraceTape;
    using namespace mecan::autograd::functional;

    TraceTape::clear();

    mecan::Tensor x_data({1, 10}, mecan::core::ScalarType::Float32);
    mecan::Tensor w0_data({10, 16}, mecan::core::ScalarType::Float32);
    mecan::Tensor b0_data({1, 16}, mecan::core::ScalarType::Float32);
    mecan::Tensor w1_data({16, 1}, mecan::core::ScalarType::Float32);
    mecan::Tensor b1_data({1, 1}, mecan::core::ScalarType::Float32);

    for (size_t i = 0; i < x_data.numel(); ++i) x_data.data_ptr<float>()[i] = 0.1f * static_cast<float>(i + 1);
    for (size_t i = 0; i < w0_data.numel(); ++i) w0_data.data_ptr<float>()[i] = 0.01f * static_cast<float>((i % 13) + 1);
    for (size_t i = 0; i < b0_data.numel(); ++i) b0_data.data_ptr<float>()[i] = 0.001f * static_cast<float>(i + 1);
    for (size_t i = 0; i < w1_data.numel(); ++i) w1_data.data_ptr<float>()[i] = 0.02f * static_cast<float>((i % 7) + 1);
    b1_data.data_ptr<float>()[0] = 0.05f;

    Variable x(x_data, false);
    Variable w0(w0_data, true);
    Variable b0(b0_data, true);
    Variable w1(w1_data, true);
    Variable b1(b1_data, true);

    TraceTape::mark_source(x, "input");
    TraceTape::mark_source(w0, "Linear[0].weight");
    TraceTape::mark_source(b0, "Linear[0].bias");
    TraceTape::mark_source(w1, "Linear[2].weight");
    TraceTape::mark_source(b1, "Linear[2].bias");

    Variable z0 = matmul(x, w0);
    Variable z1 = add(z0, b0);
    Variable h = relu(z1);
    Variable z2 = matmul(h, w1);
    Variable y = add(z2, b1);

    TraceTape::tag_node(w0.node_id, "module", "Linear[0]");
    TraceTape::tag_node(w1.node_id, "module", "Linear[2]");
    TraceTape::tag_node(h.node_id, "module", "ReLU[1]");

    y.backward();

    std::cout << "Autograd demo -> y=" << y.data.data_ptr<float>()[0]
              << " dL/dw0[0]=" << w0.grad.data_ptr<float>()[0]
              << " dL/dw1[0]=" << w1.grad.data_ptr<float>()[0] << std::endl;

    const auto full_stats = TraceTape::stats();
    const auto linear0_edges = TraceTape::subgraph_by_label("Linear[0]");
    const auto simplified = TraceTape::simplify_chains();

    std::cout << "Graph stats -> nodes=" << full_stats.node_count
              << " edges=" << full_stats.edge_count
              << " tags=" << full_stats.tag_count
              << " linear0_edges=" << linear0_edges.size()
              << " simplified_edges=" << simplified.size() << std::endl;

    std::ofstream full_dot("graph_full.dot");
    full_dot << TraceTape::to_dot();
    full_dot.close();

    std::ofstream simple_dot("graph_simplified.dot");
    simple_dot << TraceTape::to_dot(simplified, true);
    simple_dot.close();

    std::cout << "Graph DOT files written: graph_full.dot, graph_simplified.dot" << std::endl;

    return 0;
}
