#include "./../include/astar.h"
#include <queue>
#include <limits>
#include <cmath>
#include <unordered_set>

std::unordered_map<Node, Weight> astar(
    const Graph& graph, 
    Node source, 
    Node target,
    const HeuristicFunction& heuristic,
    Instrument* instr) {
    
    Instrument local_instr;
    if (!instr) instr = &local_instr;
    
    // Optimización: reservar espacio para evitar reallocaciones
    std::unordered_map<Node, Weight> g_cost;
    std::unordered_map<Node, Weight> f_cost;
    std::unordered_map<Node, Node> parent;
    
    g_cost.reserve(graph.size());
    f_cost.reserve(graph.size());
    parent.reserve(graph.size());
    
    // Inicializar costos
    for (const auto& [node, _] : graph) {
        g_cost[node] = std::numeric_limits<Weight>::infinity();
        f_cost[node] = std::numeric_limits<Weight>::infinity();
    }
    
    g_cost[source] = 0.0;
    f_cost[source] = heuristic(source, target);
    
    // Usar priority_queue con comparador personalizado para mejor rendimiento
    std::priority_queue<AStarNode, std::vector<AStarNode>, std::greater<AStarNode>> open_list;
    std::unordered_set<Node> closed_set;
    std::unordered_set<Node> open_set;  // Para verificación rápida
    
    open_list.push(AStarNode(source, g_cost[source], f_cost[source]));
    open_set.insert(source);
    instr->heap_ops++;
    
    while (!open_list.empty()) {
        AStarNode current = open_list.top();
        open_list.pop();
        instr->heap_ops++;
        
        Node u = current.node;
        
        if (u == target) {
            // Reconstruir camino y devolver distancias
            std::unordered_map<Node, Weight> dist;
            dist.reserve(graph.size());
            dist[target] = g_cost[target];
            
            Node curr = target;
            while (parent.find(curr) != parent.end()) {
                curr = parent[curr];
                dist[curr] = g_cost[curr];
            }
            
            return dist;
        }
        
        if (closed_set.find(u) != closed_set.end()) {
            continue;
        }
        
        closed_set.insert(u);
        open_set.erase(u);
        
        if (graph.find(u) != graph.end()) {
            for (const auto& [v, w] : graph.at(u)) {
                if (closed_set.find(v) != closed_set.end()) {
                    continue;
                }
                
                instr->relaxations++;
                Weight tentative_g = g_cost[u] + w;
                
                if (tentative_g < g_cost[v]) {
                    parent[v] = u;
                    g_cost[v] = tentative_g;
                    f_cost[v] = g_cost[v] + heuristic(v, target);
                    
                    // Solo agregar si no está ya en open_set
                    if (open_set.find(v) == open_set.end()) {
                        open_list.push(AStarNode(v, g_cost[v], f_cost[v]));
                        open_set.insert(v);
                        instr->heap_ops++;
                    }
                }
            }
        }
    }
    
    // Si no se encontró camino, devolver distancias parciales
    return g_cost;
}

Weight euclidean_heuristic(Node a, Node b) {
    // Heurística simple basada en diferencia de IDs
    // En un grafo real, esto sería la distancia euclidiana real
    return std::abs(a - b) * 0.1;
}

Weight zero_heuristic(Node a, Node b) {
    return 0.0;
}

Weight grid_heuristic(Node a, Node b, int cols) {
    // Heurística para grafos de malla 2D
    int ax = a % cols, ay = a / cols;
    int bx = b % cols, by = b / cols;
    return std::sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by));
}
