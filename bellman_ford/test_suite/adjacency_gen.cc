#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <utility> 
#include "adjacency_gen.h"

using namespace std;

// Function to read CSV and construct the adjacency list
vector<vector<pair<int, int>>> loadAdjacencyListFromCSV(const string& filename, int& numNodes) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file.\n";
    }

    string line;
    int maxNode = -1;
    vector<tuple<int, int, int>> edges;

    while (getline(file, line)) {
        stringstream ss(line);
        string token;
        int source, dest, weight;

        // Parse source
        if (!getline(ss, token, ',')) continue;
        source = stoi(token);

        // Parse destination
        if (!getline(ss, token, ',')) continue;
        dest = stoi(token);

        // Parse weight
        if (!getline(ss, token, ',')) continue;
        weight = stoi(token);

        edges.emplace_back(source, dest, weight);
        maxNode = max(maxNode, max(source, dest));
    }

    file.close();

    numNodes = maxNode + 1; // assuming nodes are 0-indexed
    vector<vector<pair<int, int>>> adj(numNodes);

    for (auto [u, v, w] : edges) {
        adj[u].emplace_back(v, w);
        // For undirected graph, also add:
        // adj[v].emplace_back(u, w);
    }

    return adj;
}