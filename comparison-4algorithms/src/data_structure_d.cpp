#include "./../include/data_structure_d.h"
#include <algorithm>
#include <stdexcept>

void DataStructureD::cleanup() {
    while (!heap.empty()) {
        auto [key, v] = heap.top();
        auto it = best.find(v);
        if (it == best.end() || it->second != key) {
            heap.pop();
        } else {
            break;
        }
    }
}

DataStructureD::DataStructureD(int M, double B_upper, int block_size)
    : M(std::max(1, M)), B_upper(B_upper) {
    this->block_size = (block_size > 0) ? block_size : std::max(1, M / 8);
}

void DataStructureD::insert(Node v, Weight key) {
    auto it = best.find(v);
    if (it == best.end() || key < it->second) {
        best[v] = key;
        heap.push({key, v});
    }
}

void DataStructureD::batch_prepend(const std::vector<std::pair<Node, Weight>>& pairs) {
    for (const auto& [v, key] : pairs) {
        insert(v, key);
    }
}

bool DataStructureD::empty() {
    cleanup();
    return heap.empty();
}

std::pair<Weight, std::unordered_set<Node>> DataStructureD::pull() {
    cleanup();
    if (heap.empty()) {
        throw std::runtime_error("pull from empty D");
    }
    
    Weight Bi = heap.top().first;
    std::unordered_set<Node> Si;
    
    while (!heap.empty() && (int)Si.size() < block_size) {
        auto [key, v] = heap.top();
        heap.pop();
        
        auto it = best.find(v);
        if (it != best.end() && it->second == key) {
            Si.insert(v);
            best.erase(it);
        }
    }
    
    return {Bi, Si};
}