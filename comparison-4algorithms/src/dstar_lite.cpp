#include "./../include/dstar_lite.h"
#include <queue>
#include <limits>
#include <cmath>
#include <algorithm>

DStarLite::DStarLite(const Graph& g, Node s, Node g_goal, const HeuristicFunction& h, Instrument* instr)
    : graph(g), start(s), goal(g_goal), heuristic(h), instrument(instr), km(0.0) {
    initialize();
}

void DStarLite::initialize() {
    Instrument local_instr;
    if (!instrument) instrument = &local_instr;
    
    // Optimización: reservar espacio para evitar reallocaciones
    g_cost.reserve(graph.size());
    rhs_cost.reserve(graph.size());
    h_cost.reserve(graph.size());
    
    // Inicializar costos
    for (const auto& [node, _] : graph) {
        g_cost[node] = std::numeric_limits<Weight>::infinity();
        rhs_cost[node] = std::numeric_limits<Weight>::infinity();
        h_cost[node] = heuristic(node, goal);
    }
    
    rhs_cost[goal] = 0.0;
    auto key = calculate_key(goal);
    open_list.push(DStarLiteNode(goal, key.first, rhs_cost[goal], key.first));
    open_set.insert(goal);
    instrument->heap_ops++;
}

std::pair<Weight, Weight> DStarLite::calculate_key(Node u) {
    Weight g_val = std::min(g_cost[u], rhs_cost[u]);
    return std::make_pair(g_val + h_cost[u] + km, g_val);
}

void DStarLite::update_vertex(Node u) {
    if (u != goal) {
        Weight min_rhs = std::numeric_limits<Weight>::infinity();
        
        if (graph.find(u) != graph.end()) {
            for (const auto& [v, w] : graph.at(u)) {
                instrument->relaxations++;
                Weight cost = g_cost[v] + w;
                min_rhs = std::min(min_rhs, cost);
            }
        }
        
        rhs_cost[u] = min_rhs;
    }
    
    // Remover de open_list si está presente
    if (open_set.find(u) != open_set.end()) {
        open_set.erase(u);
        // Recrear priority_queue sin el nodo u
        std::priority_queue<DStarLiteNode, std::vector<DStarLiteNode>, std::greater<DStarLiteNode>> new_queue;
        std::unordered_set<Node> new_set;
        
        while (!open_list.empty()) {
            DStarLiteNode node = open_list.top();
            open_list.pop();
            if (node.node != u) {
                new_queue.push(node);
                new_set.insert(node.node);
            }
        }
        
        open_list = std::move(new_queue);
        open_set = std::move(new_set);
    }
    
    // Agregar a open_list si g != rhs
    if (g_cost[u] != rhs_cost[u]) {
        auto key_pair = calculate_key(u);
        Weight key = key_pair.first;
        open_list.push(DStarLiteNode(u, key, rhs_cost[u], key));
        open_set.insert(u);
        instrument->heap_ops++;
    }
}

void DStarLite::compute_shortest_path() {
    while (!open_list.empty()) {
        DStarLiteNode current = open_list.top();
        open_list.pop();
        instrument->heap_ops++;
        
        Node u = current.node;
        Weight k_old = current.f_cost;
        auto k_new_pair = calculate_key(u);
        Weight k_new = k_new_pair.first;
        
        if (k_old < k_new) {
            open_list.push(DStarLiteNode(u, k_new, rhs_cost[u], k_new));
            instrument->heap_ops++;
            continue;
        }
        
        if (g_cost[u] > rhs_cost[u]) {
            g_cost[u] = rhs_cost[u];
            
            if (graph.find(u) != graph.end()) {
                for (const auto& [v, w] : graph.at(u)) {
                    update_vertex(v);
                }
            }
        } else {
            g_cost[u] = std::numeric_limits<Weight>::infinity();
            
            if (graph.find(u) != graph.end()) {
                for (const auto& [v, w] : graph.at(u)) {
                    update_vertex(v);
                }
            }
            update_vertex(u);
        }
        
        if (u == start) {
            break;
        }
    }
}

std::unordered_map<Node, Weight> DStarLite::find_path() {
    compute_shortest_path();
    
    // Reconstruir camino desde start hasta goal
    std::unordered_map<Node, Weight> dist;
    Node current = start;
    
    while (current != goal && g_cost[current] != std::numeric_limits<Weight>::infinity()) {
        dist[current] = g_cost[current];
        
        Node next = current;
        Weight min_cost = std::numeric_limits<Weight>::infinity();
        
        if (graph.find(current) != graph.end()) {
            for (const auto& [v, w] : graph.at(current)) {
                Weight cost = g_cost[v] + w;
                if (cost < min_cost) {
                    min_cost = cost;
                    next = v;
                }
            }
        }
        
        if (next == current) break; // No hay camino válido
        current = next;
    }
    
    if (current == goal) {
        dist[goal] = g_cost[goal];
    }
    
    return dist;
}

void DStarLite::update_graph(const std::vector<Edge>& changed_edges) {
    km += heuristic(start, goal);
    
    for (const auto& edge : changed_edges) {
        update_vertex(edge.from);
    }
}

std::unordered_map<Node, Weight> DStarLite::replan() {
    compute_shortest_path();
    return find_path();
}

// Función wrapper para compatibilidad
std::unordered_map<Node, Weight> dstar_lite(
    const Graph& graph, 
    Node source, 
    Node target,
    const HeuristicFunction& heuristic,
    Instrument* instr) {
    
    DStarLite planner(graph, source, target, heuristic, instr);
    return planner.find_path();
}
