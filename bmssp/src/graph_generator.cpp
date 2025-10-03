#include "./../include/graph_generator.h"
#include <random>
#include <algorithm>

std::pair<Graph, std::vector<Edge>> generate_sparse_directed_graph(
    int n, int m, double max_w, unsigned seed) {
    
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> weight_dist(1.0, max_w);
    std::uniform_int_distribution<int> node_dist(0, n - 1);
    
    Graph graph;
    std::vector<Edge> edges;
    
    for (int i = 0; i < n; ++i) {
        graph[i] = std::vector<std::pair<Node, Weight>>();
    }
    
    for (int i = 1; i < n; ++i) {
        std::uniform_int_distribution<int> prev_dist(0, i - 1);
        int u = prev_dist(rng);
        double w = weight_dist(rng);
        graph[u].push_back({i, w});
        edges.push_back({u, i, w});
    }
    
    int remaining = std::max(0, m - (n - 1));
    for (int i = 0; i < remaining; ++i) {
        int u = node_dist(rng);
        int v = node_dist(rng);
        double w = weight_dist(rng);
        graph[u].push_back({v, w});
        edges.push_back({u, v, w});
    }
    
    return {graph, edges};
}