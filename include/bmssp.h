#ifndef BMSSP_H
#define BMSSP_H

#include "types.h"
#include <unordered_set>
#include <utility>

std::pair<std::unordered_set<Node>, std::unordered_set<Node>> find_pivots(
    const Graph& graph,
    std::unordered_map<Node, Weight>& dist,
    const std::unordered_set<Node>& S,
    double B, 
    int n, 
    int k_steps, 
    int p_limit,
    Instrument* instr = nullptr
);

std::pair<double, std::unordered_set<Node>> basecase(
    const Graph& graph,
    std::unordered_map<Node, Weight>& dist,
    double B,
    const std::unordered_set<Node>& S,
    int k,
    Instrument* instr = nullptr
);

std::pair<double, std::unordered_set<Node>> bmssp(
    const Graph& graph,
    std::unordered_map<Node, Weight>& dist,
    const std::vector<Edge>& edges,
    int l, 
    double B,
    const std::unordered_set<Node>& S,
    int n,
    Instrument* instr = nullptr
);

#endif