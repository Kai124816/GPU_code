#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <utility> 
#include <chrono>
#include <hip/hip_runtime.h>
#include "adjacency_gen.h"
#include "cpu_bellman.h"
#include "../algorithm/gpu_bellman.h"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <adjacency_list.csv>\n";
        return 1;
    }

    vector<int> node_lengths = {100000, 250000, 500000, 750000}

    for(int i = 1; i < argc; i++)
    {
        string filename = argv[i];
        int numNodes = node_lengths[i];

        cout << "Loading graph from CSV: " << filename << endl;
        vector<vector<pair<int, int>>> adj_list = loadAdjacencyListFromCSV(filename, numNodes);

        cout << "Running CPU Bellman-Ford..." << endl;
        auto cpu_start = chrono::high_resolution_clock::now();
        vector<int> cpu_result = cpu_bellman(adj_list);
        auto cpu_end = chrono::high_resolution_clock::now();
        chrono::duration<double> cpu_duration = cpu_end - cpu_start;
        cout << "GPU Bellman-Ford took on graph with " << numNodes << " nodes" << cpu_duration.count() << " seconds.\n";

        cout << "Running GPU Bellman-Ford..." << endl;
        auto gpu_start = chrono::high_resolution_clock::now();
        vector<int> gpu_result = gpu_bellman(adj_list);
        auto gpu_end = chrono::high_resolution_clock::now();
        chrono::duration<double> gpu_duration = gpu_end - gpu_start;
        cout << "GPU Bellman-Ford took on graph with " << numNodes << " nodes" << gpu_duration.count() << " seconds.\n";
    }
    return 0;
}
