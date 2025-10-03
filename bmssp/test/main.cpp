#include "./../include/types.h"
#include "./../include/graph_generator.h"
#include "./../include/dijkstra.h"
#include "./../include/bmssp.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <limits>

void run_single_test(int n, int m, unsigned seed = 0, Node source = 0) {
    std::cout << "Generando grafo: n=" << n << ", m=" << m << ", seed=" << seed << "\n";
    
    auto [graph, edges] = generate_sparse_directed_graph(n, m, 100.0, seed);
    
    double avg_deg = 0.0;
    for (const auto& [node, adj] : graph) {
        avg_deg += adj.size();
    }
    avg_deg /= n;
    std::cout << "Grafo generado. grado promedio " << avg_deg << "\n";
    
    // Dijkstra
    Instrument instr_dij;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto dist_dij = dijkstra(graph, source, &instr_dij);
    auto t1 = std::chrono::high_resolution_clock::now();
    double time_dij = std::chrono::duration<double>(t1 - t0).count();
    
    int reachable_dij = 0;
    for (const auto& [_, d] : dist_dij) {
        if (std::isfinite(d)) reachable_dij++;
    }
    
    std::cout << "Dijkstra: time=" << time_dij << "s, relaxations="
              << instr_dij.relaxations << ", heap_ops=" << instr_dij.heap_ops
              << ", reachable=" << reachable_dij << "\n";
    
    std::unordered_map<Node, Weight> dist_bm;
    for (const auto& [node, _] : graph) {
        dist_bm[node] = std::numeric_limits<Weight>::infinity();
    }
    dist_bm[source] = 0.0;
    
    int l;
    if (n <= 2) {
        l = 1;
    } else {
        int t_guess = std::max(1, (int)std::round(std::pow(std::log(std::max(3, n)), 2.0 / 3.0)));
        l = std::max(1, (int)std::round(std::log(std::max(3, n)) / t_guess));
    }
    
    std::cout << "BMSSP params: top-level l=" << l << "\n";
    
    Instrument instr_bm;
    t0 = std::chrono::high_resolution_clock::now();
    auto [Bp, U_final] = bmssp(graph, dist_bm, edges, l,
                                std::numeric_limits<double>::infinity(),
                                {source}, n, &instr_bm);
    t1 = std::chrono::high_resolution_clock::now();
    double time_bm = std::chrono::duration<double>(t1 - t0).count();
    
    int reachable_bm = 0;
    for (const auto& [_, d] : dist_bm) {
        if (std::isfinite(d)) reachable_bm++;
    }
    
    std::cout << "BMSSP: time=" << time_bm << "s, relaxations="
              << instr_bm.relaxations << ", reachable=" << reachable_bm
              << ", B'=" << Bp << ", |U_final|=" << U_final.size() << "\n";
    
    double max_diff = 0.0;
    for (const auto& [v, dv] : dist_dij) {
        double db = dist_bm[v];
        if (std::isfinite(dv) && std::isfinite(db)) {
            max_diff = std::max(max_diff, std::abs(dv - db));
        }
    }
    
    std::cout << "Diferencia máxima en distancias: " << max_diff << "\n";
}

int main(int argc, char* argv[]) {
    int n = 200000;
    int m = 800000;
    unsigned seed = 0;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-n" || arg == "--nodes") && i + 1 < argc) {
            n = std::atoi(argv[++i]);
        } else if ((arg == "-m" || arg == "--edges") && i + 1 < argc) {
            m = std::atoi(argv[++i]);
        } else if ((arg == "-s" || arg == "--seed") && i + 1 < argc) {
            seed = std::atoi(argv[++i]);
        }
    }
    
    run_single_test(n, m, seed);
    return 0;
}