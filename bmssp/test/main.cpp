#include "./../include/types.h"
#include "./../include/graph_generator.h"
#include "./../include/dijkstra.h"
#include "./../include/bmssp.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>

struct BenchResult {
    double time_dij;
    double time_bm;
};

BenchResult run_single_benchmark(int n, int m, unsigned seed = 0, Node source = 0) {
    auto [graph, edges] = generate_sparse_directed_graph(n, m, 100.0, seed);

    Instrument instr_dij;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto dist_dij = dijkstra(graph, source, &instr_dij);
    auto t1 = std::chrono::high_resolution_clock::now();
    double time_dij = std::chrono::duration<double>(t1 - t0).count();

    std::unordered_map<Node, Weight> dist_bm;
    dist_bm.reserve(graph.size());
    for (const auto& [node, _] : graph) {
        dist_bm[node] = std::numeric_limits<Weight>::infinity();
    }
    dist_bm[source] = 0.0;

    int l;
    int n_nodes = (int)graph.size();
    if (n_nodes <= 2) {
        l = 1;
    } else {
        int t_guess = std::max(1, (int)std::round(std::pow(std::log(std::max(3, n_nodes)), 2.0 / 3.0)));
        l = std::max(1, (int)std::round(std::log(std::max(3, n_nodes)) / t_guess));
    }

    Instrument instr_bm;
    t0 = std::chrono::high_resolution_clock::now();
    auto [Bp, U_final] = bmssp(
        graph, dist_bm, edges, l,
        std::numeric_limits<double>::infinity(),
        {source}, n_nodes, &instr_bm
    );
    t1 = std::chrono::high_resolution_clock::now();
    double time_bm = std::chrono::duration<double>(t1 - t0).count();

    return {time_dij, time_bm};
}

int main(int argc, char* argv[]) {
    int n = 200000;
    int m = 800000;
    unsigned seed0 = 0;
    int trials = 100;
    Node source = 0;
    std::string out_path = "benchmark_times.csv";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-n" || arg == "--nodes") && i + 1 < argc) {
            n = std::atoi(argv[++i]);
        } else if ((arg == "-m" || arg == "--edges") && i + 1 < argc) {
            m = std::atoi(argv[++i]);
        } else if ((arg == "-s" || arg == "--seed") && i + 1 < argc) {
            seed0 = std::atoi(argv[++i]);
        } else if ((arg == "-t" || arg == "--trials") && i + 1 < argc) {
            trials = std::atoi(argv[++i]);
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            out_path = argv[++i];
        } else if ((arg == "--source") && i + 1 < argc) {
            source = static_cast<Node>(std::atoi(argv[++i]));
        }
    }

    std::ofstream fout(out_path);
    if (!fout) {
        std::cerr << "Error: no se pudo abrir el archivo de salida: " << out_path << "\n";
        return 1;
    }

    fout << "trial,seed,time_dijkstra,time_bmssp\n";

    for (int i = 0; i < trials; ++i) {
        unsigned seed = seed0 + static_cast<unsigned>(i);
        BenchResult r = run_single_benchmark(n, m, seed, source);
        fout << i << "," << seed << "," << r.time_dij << "," << r.time_bm << "\n";
    }

    fout.close();
    std::cout << "Output listo " << trials << " test => " << out_path << "\n";
    return 0;
}
