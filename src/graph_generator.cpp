#include "./../include/graph_generator.h"
#include <random>
#include <algorithm>
#include <unordered_set>
#include <numeric>

static inline void ensure_node(Graph& G, Node u) {
    if (!G.count(u)) G[u] = std::vector<std::pair<Node, Weight>>{};
}
static inline void add_edge(Graph& G, std::vector<Edge>& E, Node u, Node v, Weight w) {
    ensure_node(G,u); ensure_node(G,v);
    G[u].push_back({v,w});
    E.push_back({u,v,w});
}

std::pair<Graph, std::vector<Edge>> generate_sparse_directed_graph(
    int n, int m, double max_w, unsigned seed) {
    
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> weight_dist(1.0, max_w);
    std::uniform_int_distribution<int> node_dist(0, n - 1);
    
    Graph graph;
    std::vector<Edge> edges;
    
    for (int i = 0; i < n; ++i) {
        graph[i] = std::vector<std::pair<Node, Weight>>();
    }
    

    for (int i = 1; i < n; ++i) {
        std::uniform_int_distribution<int> prev_dist(0, i - 1);
        int u = prev_dist(rng);
        double w = weight_dist(rng);
        graph[u].push_back({i, w});
        edges.push_back({u, i, w});
    }
    
    int remaining = std::max(0, m - (n - 1));
    for (int i = 0; i < remaining; ++i) {
        int u = node_dist(rng);
        int v = node_dist(rng);
        double w = weight_dist(rng);
        graph[u].push_back({v, w});
        edges.push_back({u, v, w});
    }
    
    return {graph, edges};
}

std::pair<Graph, std::vector<Edge>> generate_er_directed(
    int n, double p, double max_w, unsigned seed) 
{
    Graph G; std::vector<Edge> E; G.reserve(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0.0,1.0);
    std::uniform_real_distribution<double> W(1.0, max_w);

    for (int u=0; u<n; ++u) ensure_node(G,u);
    for (int u=0; u<n; ++u) for (int v=0; v<n; ++v) if (u!=v)
        if (U(rng) < p) add_edge(G,E,u,v,W(rng));
    return {G,E};
}

std::pair<Graph, std::vector<Edge>> generate_ba_directed(
    int n, int attach, double max_w, unsigned seed) 
{
    Graph G; std::vector<Edge> E; G.reserve(n);
    if (n<=0) return {G,E};

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> W(1.0, max_w);


    auto add = [&](int u, int v) {
        add_edge(G, E, u, v, W(rng));
    };


    int init = std::min(n, std::max(2, attach+1));
    for (int u=0; u<init; ++u) ensure_node(G,u);
    for (int u=0; u<init; ++u)
        for (int v=0; v<init; ++v)
            if (u!=v) add(u, v);


    std::vector<int> targets; 
    targets.reserve((int)E.size()*2);

    for (const auto& ed : E) {
        targets.push_back(ed.from);
        targets.push_back(ed.to);  
    }

    std::uniform_int_distribution<int> R(0, std::max(0,(int)targets.size()-1));

    for (int u = init; u < n; ++u) {
        ensure_node(G,u);
        std::unordered_set<int> chosen;
        for (int k=0; k<attach && !targets.empty(); ++k) {
            int v = targets[R(rng)];
        
            int guard = 0;
            while ((v == u || chosen.count(v)) && guard++ < 32) {
                v = targets[R(rng)];
            }
            chosen.insert(v);
        }
        for (int v : chosen) {
            add(u, v);
        
            targets.push_back(u);
            targets.push_back(v);
        }
    }
    return {G,E};
}

std::pair<Graph, std::vector<Edge>> generate_ws_directed(
    int n, int k, double beta, double max_w, unsigned seed)
{
    Graph G; std::vector<Edge> E; G.reserve(n);
    if (n<=0) return {G,E};

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0.0,1.0);
    std::uniform_real_distribution<double> W(1.0, max_w);
    std::uniform_int_distribution<int> R(0, n-1);

    if (k%2) ++k; k = std::min(k, n-1);
    for (int u=0; u<n; ++u) ensure_node(G,u);

    for (int u=0; u<n; ++u) {
        for (int d=1; d<=k/2; ++d) {
            int v = (u + d) % n;
            int w = (u - d + n) % n;

            int to = v;
            if (U(rng) < beta) { do { to = R(rng); } while (to==u); }
            add_edge(G,E,u,to,W(rng));

            to = w;
            if (U(rng) < beta) { do { to = R(rng); } while (to==u); }
            add_edge(G,E,u,to,W(rng));
        }
    }
    return {G,E};
}

std::pair<Graph, std::vector<Edge>> generate_grid2d_directed(
    int rows, int cols, bool diag, double max_w, unsigned seed)
{
    Graph G; std::vector<Edge> E; G.reserve(rows*cols);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> W(1.0, max_w);

    auto id = [cols](int r,int c){ return r*cols + c; };
    auto inside = [rows,cols](int r,int c){ return r>=0 && r<rows && c>=0 && c<cols; };

    const int dr4[4]={1,-1,0,0};
    const int dc4[4]={0,0,1,-1};
    const int dr8[8]={1,1,1,0,0,-1,-1,-1};
    const int dc8[8]={1,0,-1,1,-1,1,0,-1};

    for (int r=0;r<rows;++r) for (int c=0;c<cols;++c) {
        Node u = id(r,c); ensure_node(G,u);
        if (!diag) {
            for (int t=0;t<4;++t){
                int rr=r+dr4[t], cc=c+dc4[t];
                if (inside(rr,cc)) add_edge(G,E,u,id(rr,cc),W(rng));
            }
        } else {
            for (int t=0;t<8;++t){
                int rr=r+dr8[t], cc=c+dc8[t];
                if (inside(rr,cc)) add_edge(G,E,u,id(rr,cc),W(rng));
            }
        }
    }
    return {G,E};
}

std::pair<Graph, std::vector<Edge>> generate_layered_dag(
    int layers, int width, double p_forward, double max_w, unsigned seed)
{
    Graph G; std::vector<Edge> E;
    if (layers<=0 || width<=0) return {G,E};

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0.0,1.0);
    std::uniform_real_distribution<double> W(1.0, max_w);

    int n = layers*width;
    for (int u=0; u<n; ++u) ensure_node(G,u);

    auto nid = [width](int L, int i){ return L*width + i; };

    for (int L=0; L<layers-1; ++L) {
        for (int i=0; i<width; ++i) {
            Node u = nid(L,i);
            for (int L2=L+1; L2<layers; ++L2) {
                double p = (L2==L+1 ? p_forward : p_forward*0.3);
                for (int j=0; j<width; ++j) {
                    if (U(rng) < p) add_edge(G,E,u,nid(L2,j),W(rng));
                }
            }
        }
    }
    return {G,E};
}

std::pair<Graph, std::vector<Edge>>
generate_graph(GraphType type, const GraphGenOptions& opt) {
    switch (type) {
        case GraphType::RANDOM_M:
            return generate_sparse_directed_graph(opt.n, opt.m, opt.wmax, opt.seed);
        case GraphType::ER:
            return generate_er_directed(opt.n, opt.p, opt.wmax, opt.seed);
        case GraphType::BA:
            return generate_ba_directed(opt.n, opt.attach, opt.wmax, opt.seed);
        case GraphType::WS:
            return generate_ws_directed(opt.n, opt.k, opt.beta, opt.wmax, opt.seed);
        case GraphType::GRID2D:
            return generate_grid2d_directed(opt.rows, opt.cols, opt.diag, opt.wmax, opt.seed);
        case GraphType::LAYERED_DAG:
            return generate_layered_dag(opt.layers, opt.width, opt.dagp, opt.wmax, opt.seed);
        default:
            return generate_sparse_directed_graph(opt.n, opt.m, opt.wmax, opt.seed);
    }
}
