#ifndef DSTAR_LITE_H
#define DSTAR_LITE_H

#include "types.h"
#include <unordered_map>
#include <vector>
#include <unordered_set>
#include <queue>

// Algoritmo D*-lite para planificación dinámica
class DStarLite {
private:
    const Graph& graph;
    Node start, goal;
    HeuristicFunction heuristic;
    Instrument* instrument;
    
    std::unordered_map<Node, Weight> g_cost;
    std::unordered_map<Node, Weight> rhs_cost;
    std::unordered_map<Node, Weight> h_cost;
    
    std::priority_queue<DStarLiteNode, std::vector<DStarLiteNode>, std::greater<DStarLiteNode>> open_list;
    std::unordered_set<Node> open_set;
    
    Weight km;  // key modifier
    
    void initialize();
    void update_vertex(Node u);
    void compute_shortest_path();
    std::pair<Weight, Weight> calculate_key(Node u);
    
public:
    DStarLite(const Graph& g, Node s, Node g_goal, const HeuristicFunction& h, Instrument* instr = nullptr);
    
    // Encuentra el camino inicial
    std::unordered_map<Node, Weight> find_path();
    
    // Actualiza el grafo cuando cambian los costos de las aristas
    void update_graph(const std::vector<Edge>& changed_edges);
    
    // Recalcula el camino después de cambios
    std::unordered_map<Node, Weight> replan();
};

// Función wrapper para compatibilidad con el benchmark
std::unordered_map<Node, Weight> dstar_lite(
    const Graph& graph, 
    Node source, 
    Node target,
    const HeuristicFunction& heuristic,
    Instrument* instr = nullptr
);

#endif
