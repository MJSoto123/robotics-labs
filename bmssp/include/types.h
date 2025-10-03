#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <unordered_map>

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

#endif