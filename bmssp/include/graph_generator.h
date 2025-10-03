#ifndef GRAPH_GENERATOR_H
#define GRAPH_GENERATOR_H

#include "types.h"
#include <utility>
#include <vector>
#include <unordered_map>
#include <string>

using AdjList = std::vector<std::pair<Node, Weight>>;
using Graph   = std::unordered_map<Node, AdjList>;

std::pair<Graph, std::vector<Edge>> generate_sparse_directed_graph(
    int n,
    int m,
    double max_w = 100.0,
    unsigned seed = 0
);

enum class GraphType {
    RANDOM_M,  
    ER,        
    BA,        
    WS,        
    GRID2D,    
    LAYERED_DAG
};

struct GraphGenOptions {
    double wmax = 100.0;
    unsigned seed = 0;

    int n = 0;
    int m = 0;
    double p = 0.0;
    int attach = 2;
    int k = 6;
    double beta = 0.1;
    int rows = 0, cols = 0;
    bool diag = false;
    int layers = 0, width = 0;
    double dagp = 0.02;
};

std::pair<Graph, std::vector<Edge>> generate_graph(
    GraphType type, 
    const GraphGenOptions& opt
);

std::pair<Graph, std::vector<Edge>> generate_er_directed(
    int n, 
    double p, 
    double max_w = 100.0, 
    unsigned seed = 0
);

std::pair<Graph, std::vector<Edge>> generate_ba_directed(
    int n, 
    int attach, 
    double max_w = 100.0, 
    unsigned seed = 0
);

std::pair<Graph, std::vector<Edge>> generate_ws_directed(
    int n, 
    int k, 
    double beta, 
    double max_w = 100.0, 
    unsigned seed = 0
);

std::pair<Graph, std::vector<Edge>> generate_grid2d_directed(
    int rows, 
    int cols, 
    bool diag=false, 
    double max_w = 100.0, 
    unsigned seed = 0
);

std::pair<Graph, std::vector<Edge>> generate_layered_dag(
    int layers, 
    int width, 
    double p_forward=0.02, 
    double max_w = 100.0, 
    unsigned seed = 0
);

#endif
