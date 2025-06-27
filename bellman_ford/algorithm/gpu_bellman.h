#ifndef GPU_BELLMAN_H
#define GPU_BELLMAN_H

#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <hip/hip_runtime.h>

using namespace std;

const int INF = 1e9;

int count_edges(const vector<vector<pair<int, int>>>& adj_list);
vector<int> ideal_dimensions(int num_threads);
void initialize_data(const vector<vector<pair<int, int>>>& adj_list, int* edges_u, int* edges_v, int* edges_weight, 
int* active, int* dist, int num_nodes);
__global__ void modify_edge(int* edges_u, int* edges_v, int* edges_weight, int* active, int* dist, int num_edges);
__global__ void modify_active_edges(int* active, int* modified, int num_nodes);
vector<int> gpu_bellman(const vector<vector<pair<int, int>>>& adj_list);

#endif