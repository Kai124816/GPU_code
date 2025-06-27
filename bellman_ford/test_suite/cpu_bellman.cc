#include <vector>
#include <limits>
#include <utility>
#include "cpu_bellman.h"

using namespace std;

vector<int> cpu_bellman(const vector<vector<pair<int, int>>>& adj_list) {
    int num_nodes = adj_list.size();
    vector<int> dist(num_nodes, INF1);
    dist[0] = 0; // Assuming node 0 is the source

    // Relax edges up to (num_nodes - 1) times
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
        if (!updated) break; // Early exit if no updates
    }

    return dist;
}