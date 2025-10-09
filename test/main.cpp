#include "./../include/types.h"
#include "./../include/graph_generator.h"
#include "./../include/dijkstra.h"
#include "./../include/bmssp.h"
#include "./../include/astar.h"
#include "./../include/dstar_lite.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <random>

struct BenchResult { 
    double time_dij; 
    double time_bm; 
    double time_astar;
    double time_dstar;
};

static BenchResult run_benchmark(const Graph& G, const std::vector<Edge>& E, Node source, Node target) {
    // Dijkstra
    Instrument instr_dij;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto dist_dij = dijkstra(G, source, &instr_dij);
    auto t1 = std::chrono::high_resolution_clock::now();
    double time_dij = std::chrono::duration<double>(t1 - t0).count();

    // BMSSP
    std::unordered_map<Node, Weight> dist_bm;
    dist_bm.reserve(G.size());
    for (const auto& [u, _] : G) dist_bm[u] = std::numeric_limits<Weight>::infinity();
    dist_bm[source] = 0.0;

    int n_nodes = (int)G.size();
    int l;
    if (n_nodes <= 2) l = 1;
    else {
        int t_guess = std::max(1, (int)std::round(std::pow(std::log(std::max(3, n_nodes)), 2.0/3.0)));
        l = std::max(1, (int)std::round(std::log(std::max(3, n_nodes)) / t_guess));
    }

    Instrument instr_bm;
    t0 = std::chrono::high_resolution_clock::now();
    auto [Bp, U_final] = bmssp(G, dist_bm, E, l,
                               std::numeric_limits<double>::infinity(),
                               {source}, n_nodes, &instr_bm);
    t1 = std::chrono::high_resolution_clock::now();
    double time_bm = std::chrono::duration<double>(t1 - t0).count();

    // A*
    Instrument instr_astar;
    t0 = std::chrono::high_resolution_clock::now();
    auto dist_astar = astar(G, source, target, euclidean_heuristic, &instr_astar);
    t1 = std::chrono::high_resolution_clock::now();
    double time_astar = std::chrono::duration<double>(t1 - t0).count();

    // D*-lite
    Instrument instr_dstar;
    t0 = std::chrono::high_resolution_clock::now();
    auto dist_dstar = dstar_lite(G, source, target, euclidean_heuristic, &instr_dstar);
    t1 = std::chrono::high_resolution_clock::now();
    double time_dstar = std::chrono::duration<double>(t1 - t0).count();

    return {time_dij, time_bm, time_astar, time_dstar};
}

static GraphType parse_graph_type(const std::string& s) {
    if (s=="random-m")   return GraphType::RANDOM_M;
    if (s=="er")         return GraphType::ER;
    if (s=="ba")         return GraphType::BA;
    if (s=="ws")         return GraphType::WS;
    if (s=="grid2d")     return GraphType::GRID2D;
    if (s=="layered-dag")return GraphType::LAYERED_DAG;
    return GraphType::RANDOM_M;
}

int main(int argc, char* argv[]) {
    // defaults
    int n = 2000000, m = 8000000;  // Configurado para 2-5 millones de nodos
    unsigned seed0 = 0;
    int trials = 10;  // Reducido para pruebas grandes
    Node source = 0;
    Node target = 1000;  // Nodo objetivo para A* y D*-lite
    double wmax = 100.0;
    std::string out_path = "benchmark_times.csv";
    std::string gtype_str = "random-m";

    // specific
    double p = 0.0005;
    int attach = 2;
    int k = 6; double beta = 0.1;
    int rows = 200, cols = 200; bool diag = false;
    int layers = 50, width = 4000; double dagp = 0.02;

    // CLI parse
    for (int i=1; i<argc; ++i) {
        std::string a = argv[i];
        auto need = [&](int more){ return i+more < argc; };
        if ((a=="-n"||a=="--nodes") && need(1)) n = std::atoi(argv[++i]);
        else if ((a=="-m"||a=="--edges") && need(1)) m = std::atoi(argv[++i]);
        else if ((a=="-s"||a=="--seed") && need(1)) seed0 = std::atoi(argv[++i]);
        else if ((a=="-t"||a=="--trials") && need(1)) trials = std::atoi(argv[++i]);
        else if ((a=="-o"||a=="--output") && need(1)) out_path = argv[++i];
        else if ((a=="--source") && need(1)) source = static_cast<Node>(std::atoi(argv[++i]));
        else if ((a=="--target") && need(1)) target = static_cast<Node>(std::atoi(argv[++i]));
        else if ((a=="--wmax") && need(1)) wmax = std::atof(argv[++i]);
        else if ((a=="--graph") && need(1)) gtype_str = argv[++i];
        // específicos
        else if ((a=="--p") && need(1)) p = std::atof(argv[++i]);
        else if ((a=="--attach") && need(1)) attach = std::atoi(argv[++i]);
        else if ((a=="--k") && need(1)) k = std::atoi(argv[++i]);
        else if ((a=="--beta") && need(1)) beta = std::atof(argv[++i]);
        else if ((a=="--rows") && need(1)) rows = std::atoi(argv[++i]);
        else if ((a=="--cols") && need(1)) cols = std::atoi(argv[++i]);
        else if (a=="--diag") diag = true;
        else if ((a=="--layers") && need(1)) layers = std::atoi(argv[++i]);
        else if ((a=="--width") && need(1)) width = std::atoi(argv[++i]);
        else if ((a=="--dagp") && need(1)) dagp = std::atof(argv[++i]);
    }

    GraphType gtype = parse_graph_type(gtype_str);

    std::ofstream fout(out_path);
    if (!fout) {
        std::cerr << "Error: cannot open output file: " << out_path << "\n";
        return 1;
    }
    fout << "trial,seed,time_dijkstra,time_bmssp,time_astar,time_dstar_lite\n";

    for (int i=0; i<trials; ++i) {
        GraphGenOptions opt;
        opt.wmax = wmax;
        opt.seed = seed0 + (unsigned)i;
        switch (gtype) {
            case GraphType::RANDOM_M:  opt.n = n; opt.m = m; break;
            case GraphType::ER:        opt.n = n; opt.p = p; break;
            case GraphType::BA:        opt.n = n; opt.attach = attach; break;
            case GraphType::WS:        opt.n = n; opt.k = k; opt.beta = beta; break;
            case GraphType::GRID2D:    opt.rows = rows; opt.cols = cols; opt.diag = diag; break;
            case GraphType::LAYERED_DAG: opt.layers = layers; opt.width = width; opt.dagp = dagp; break;
        }

        auto [G, E] = generate_graph(gtype, opt);
        if (!G.count(source)) source = 0;
        if (!G.count(target)) target = std::min(n-1, 1000);  // Asegurar que target existe

        BenchResult r = run_benchmark(G, E, source, target);
        fout << i << "," << opt.seed << "," << r.time_dij << "," << r.time_bm << "," << r.time_astar << "," << r.time_dstar << "\n";
    }

    fout.close();
    std::cout << "CSV listo ("<<trials<<" tests) => " << out_path << "\n";
    return 0;
}
