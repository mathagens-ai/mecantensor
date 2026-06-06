#pragma once

#include <atomic>
#include <cstdint>
#include <algorithm>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "variable.h"

namespace mecan {
namespace autograd {

struct GraphEdge {
    uint64_t from_id;
    uint64_t to_id;
    std::string from_label;
    std::string to_label;
    std::string op_name;
};

struct GraphTag {
    uint64_t node_id;
    std::string level;   // info, debug, warn, error
    std::string kind;    // text, image, plot, tensor_stats
    std::string payload;
};

struct GraphStats {
    size_t node_count = 0;
    size_t edge_count = 0;
    size_t tag_count = 0;
};

class TraceTape {
public:
    struct PreCallContext {
        std::string op_name;
        std::vector<std::pair<uint64_t, std::string>> inputs;
    };

    static void mark_source(Variable& v, const std::string& label) {
        if (v.node_id == 0) {
            v.node_id = next_node_id();
        }
        v.source_node = label;
    }

    static PreCallContext pre_call(const std::string& op_name, const std::vector<const Variable*>& inputs) {
        PreCallContext ctx;
        ctx.op_name = op_name;
        ctx.inputs.reserve(inputs.size());

        for (const Variable* src : inputs) {
            if (!src) {
                continue;
            }

            // Source marking can arrive from const call sites.
            Variable* mutable_src = const_cast<Variable*>(src);
            if (mutable_src->node_id == 0) {
                mutable_src->node_id = next_node_id();
            }
            if (mutable_src->source_node.empty()) {
                mutable_src->source_node = "input";
            }

            ctx.inputs.emplace_back(mutable_src->node_id, mutable_src->source_node);
        }
        return ctx;
    }

    static void post_call(const PreCallContext& ctx, Variable& output) {
        if (output.node_id == 0) {
            output.node_id = next_node_id();
        }
        if (output.source_node.empty() || output.source_node == "input") {
            output.source_node = ctx.op_name;
        }

        std::lock_guard<std::mutex> lock(edge_mutex());
        std::vector<GraphEdge>& edges = graph_edges();
        for (const auto& in : ctx.inputs) {
            edges.push_back(GraphEdge{
                in.first,
                output.node_id,
                in.second,
                output.source_node,
                ctx.op_name
            });
        }
    }

    static std::vector<GraphEdge> edges_snapshot() {
        std::lock_guard<std::mutex> lock(edge_mutex());
        return graph_edges();
    }

    static void tag_node(uint64_t node_id,
                         const std::string& kind,
                         const std::string& payload,
                         const std::string& level = "info") {
        std::lock_guard<std::mutex> lock(tag_mutex());
        graph_tags().push_back(GraphTag{node_id, level, kind, payload});
    }

    static std::vector<GraphTag> tags_snapshot() {
        std::lock_guard<std::mutex> lock(tag_mutex());
        return graph_tags();
    }

    static GraphStats stats() {
        GraphStats s{};
        const std::vector<GraphEdge> edges = edges_snapshot();
        std::unordered_map<uint64_t, bool> nodes;
        for (const auto& e : edges) {
            nodes[e.from_id] = true;
            nodes[e.to_id] = true;
        }
        s.node_count = nodes.size();
        s.edge_count = edges.size();
        s.tag_count = tags_snapshot().size();
        return s;
    }

    static std::vector<GraphEdge> subgraph_by_label(const std::string& label_fragment) {
        const std::vector<GraphEdge> edges = edges_snapshot();
        std::vector<GraphEdge> out;
        out.reserve(edges.size());
        for (const auto& e : edges) {
            if (e.from_label.find(label_fragment) != std::string::npos ||
                e.to_label.find(label_fragment) != std::string::npos ||
                e.op_name.find(label_fragment) != std::string::npos) {
                out.push_back(e);
            }
        }
        return out;
    }

    static std::vector<GraphEdge> simplify_chains() {
        return simplify_chains(edges_snapshot());
    }

    static std::vector<GraphEdge> simplify_chains(const std::vector<GraphEdge>& source_edges) {
        if (source_edges.empty()) {
            return {};
        }
        std::vector<GraphEdge> edges = source_edges;

        bool changed = true;
        while (changed) {
            changed = false;
            std::unordered_map<uint64_t, size_t> indegree;
            std::unordered_map<uint64_t, size_t> outdegree;
            std::unordered_map<uint64_t, size_t> in_idx;
            std::unordered_map<uint64_t, size_t> out_idx;

            for (size_t i = 0; i < edges.size(); ++i) {
                indegree[edges[i].to_id] += 1;
                outdegree[edges[i].from_id] += 1;
                in_idx[edges[i].to_id] = i;
                out_idx[edges[i].from_id] = i;
            }

            for (const auto& kv : indegree) {
                const uint64_t node = kv.first;
                if (kv.second != 1 || outdegree[node] != 1) {
                    continue;
                }

                const size_t e_in = in_idx[node];
                const size_t e_out = out_idx[node];
                if (e_in == e_out || e_in >= edges.size() || e_out >= edges.size()) {
                    continue;
                }

                GraphEdge merged{
                    edges[e_in].from_id,
                    edges[e_out].to_id,
                    edges[e_in].from_label,
                    edges[e_out].to_label,
                    edges[e_in].op_name + "+" + edges[e_out].op_name
                };

                std::vector<GraphEdge> next;
                next.reserve(edges.size());
                for (size_t i = 0; i < edges.size(); ++i) {
                    if (i == e_in || i == e_out) continue;
                    next.push_back(edges[i]);
                }
                next.push_back(merged);
                edges.swap(next);
                changed = true;
                break;
            }
        }
        return edges;
    }

    static void clear() {
        {
            std::lock_guard<std::mutex> lock(edge_mutex());
            graph_edges().clear();
        }
        {
            std::lock_guard<std::mutex> lock(tag_mutex());
            graph_tags().clear();
        }
    }

    static std::string to_dot() {
        return to_dot(edges_snapshot(), true);
    }

    static std::string to_dot(const std::vector<GraphEdge>& edges, bool include_tags) {
        std::ostringstream os;
        os << "digraph TSTensorGraph {\n";
        os << "  rankdir=LR;\n";
        for (const auto& e : edges) {
            os << "  n" << e.from_id << " [label=\"" << e.from_label << "\"];\n";
            os << "  n" << e.to_id << " [label=\"" << e.to_label << "\"];\n";
            os << "  n" << e.from_id << " -> n" << e.to_id << " [label=\"" << e.op_name << "\"];\n";
        }
        if (include_tags) {
            const std::vector<GraphTag> tags = tags_snapshot();
            for (const auto& t : tags) {
                os << "  n" << t.node_id << " [xlabel=\"" << t.kind << ":" << t.payload << "\"];\n";
            }
        }
        os << "}\n";
        return os.str();
    }

private:
    static std::vector<GraphEdge>& graph_edges() {
        static std::vector<GraphEdge> edges;
        return edges;
    }

    static std::vector<GraphTag>& graph_tags() {
        static std::vector<GraphTag> tags;
        return tags;
    }

    static std::mutex& edge_mutex() {
        static std::mutex m;
        return m;
    }

    static std::mutex& tag_mutex() {
        static std::mutex m;
        return m;
    }

    static uint64_t next_node_id() {
        static std::atomic<uint64_t> counter{1};
        return counter.fetch_add(1, std::memory_order_relaxed);
    }
};

} // namespace autograd
} // namespace mecan
