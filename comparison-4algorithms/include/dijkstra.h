#ifndef DIJKSTRA_H
#define DIJKSTRA_H

#include "types.h"
#include <unordered_map>

std::unordered_map<Node, Weight> dijkstra(
    const Graph& graph, 
    Node source, 
    Instrument* instr = nullptr
);

#endif