#include "./../include/dijkstra.h"
#include <queue>
#include <limits>

std::unordered_map<Node, Weight> dijkstra(
    const Graph& graph, Node source, Instrument* instr) {
    
    Instrument local_instr;
    if (!instr) instr = &local_instr;
    
    std::unordered_map<Node, Weight> dist;
    for (const auto& [node, _] : graph) {
        dist[node] = std::numeric_limits<Weight>::infinity();
    }
    dist[source] = 0.0;
    
    using PQPair = std::pair<Weight, Node>;
    std::priority_queue<PQPair, std::vector<PQPair>, std::greater<PQPair>> heap;
    
    heap.push({0.0, source});
    instr->heap_ops++;
    
    while (!heap.empty()) {
        auto [d_u, u] = heap.top();
        heap.pop();
        instr->heap_ops++;
        
        if (d_u > dist[u]) continue;
        
        if (graph.find(u) != graph.end()) {
            for (const auto& [v, w] : graph.at(u)) {
                instr->relaxations++;
                Weight alt = d_u + w;
                if (alt < dist[v]) {
                    dist[v] = alt;
                    heap.push({alt, v});
                    instr->heap_ops++;
                }
            }
        }
    }
    
    return dist;
}