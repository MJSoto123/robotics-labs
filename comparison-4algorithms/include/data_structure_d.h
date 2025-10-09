#ifndef DATA_STRUCTURE_D_H
#define DATA_STRUCTURE_D_H

#include "types.h"
#include <queue>
#include <unordered_set>
#include <utility>

class DataStructureD {
private:
    using PQPair = std::pair<Weight, Node>;
    std::priority_queue<PQPair, std::vector<PQPair>, std::greater<PQPair>> heap;
    std::unordered_map<Node, Weight> best;
    int M;
    double B_upper;
    int block_size;
    
    void cleanup();
    
public:
    DataStructureD(int M, double B_upper, int block_size = -1);
    
    void insert(Node v, Weight key);
    void batch_prepend(const std::vector<std::pair<Node, Weight>>& pairs);
    bool empty();
    std::pair<Weight, std::unordered_set<Node>> pull();
};

#endif