#include <iostream>
#include <vector>
#include <cerrno>
#include <limits>
#include <utility>
#include "cpu_bellman.h"

using namespace std;


vector<int> cpu_bellman(const vector<vector<pair<int, int>>>& adj_list) {
    int num_nodes = adj_list.size();
    vector<int> dist(num_nodes, INF1);
    dist[0] = 0; // Source node

    // Main relaxation loop
    for (int i = 0; i < num_nodes - 1; ++i) {
        bool updated = false;
        for (int u = 0; u < num_nodes; ++u) {
            if (dist[u] == INF1) continue;
            for (const auto& [v, w] : adj_list[u]) {
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    updated = true;
                }
            }
        }
        if (!updated) break; // Early termination
    }

    // Negative cycle detection
    for (int u = 0; u < num_nodes; ++u) {
        if (dist[u] == INF1) continue;
        for (const auto& [v, w] : adj_list[u]) {
            if (dist[u] + w < dist[v]) {
                std::cerr << "Error: Negative-weight cycle detected.\n";
                return {};  // Or handle differently
            }
        }
    }

    return dist;
}
