#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <unordered_map>
#include <functional>

using Node = int;
using Weight = double;

struct Edge {
    Node from, to;
    Weight weight;
};

using Graph = std::unordered_map<Node, std::vector<std::pair<Node, Weight>>>;

struct Instrument {
    size_t relaxations = 0;
    size_t heap_ops = 0;
    
    void reset() {
        relaxations = 0;
        heap_ops = 0;
    }
};

// Estructuras para A* y D*-lite
struct AStarNode {
    Node node;
    Weight g_cost;  // costo real desde inicio
    Weight f_cost;  // g_cost + h_cost (heurística)
    
    AStarNode(Node n, Weight g, Weight f) : node(n), g_cost(g), f_cost(f) {}
    
    bool operator>(const AStarNode& other) const {
        return f_cost > other.f_cost;
    }
};

struct DStarLiteNode {
    Node node;
    Weight g_cost;
    Weight rhs_cost;  // right-hand-side cost
    Weight f_cost;
    
    DStarLiteNode(Node n, Weight g, Weight rhs, Weight f) 
        : node(n), g_cost(g), rhs_cost(rhs), f_cost(f) {}
    
    bool operator>(const DStarLiteNode& other) const {
        if (f_cost != other.f_cost) return f_cost > other.f_cost;
        return g_cost > other.g_cost;
    }
};

// Función heurística (distancia euclidiana simplificada para grafos)
using HeuristicFunction = std::function<Weight(Node, Node)>;

#endif