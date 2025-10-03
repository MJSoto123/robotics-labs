#ifndef GRAPH_GENERATOR_H
#define GRAPH_GENERATOR_H

#include "types.h"
#include <utility>

std::pair<Graph, std::vector<Edge>> generate_sparse_directed_graph(
    int n, 
    int m, 
    double max_w = 100.0, 
    unsigned seed = 0
);

#endif