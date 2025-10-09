#ifndef ASTAR_H
#define ASTAR_H

#include "types.h"
#include <unordered_map>
#include <vector>

// Algoritmo A* para encontrar camino más corto desde source hasta target
std::unordered_map<Node, Weight> astar(
    const Graph& graph, 
    Node source, 
    Node target,
    const HeuristicFunction& heuristic,
    Instrument* instr = nullptr
);

// Heurística simple basada en distancia euclidiana estimada
Weight euclidean_heuristic(Node a, Node b);

// Heurística cero (equivale a Dijkstra)
Weight zero_heuristic(Node a, Node b);

// Heurística para grafos de malla 2D
Weight grid_heuristic(Node a, Node b, int cols);

#endif

