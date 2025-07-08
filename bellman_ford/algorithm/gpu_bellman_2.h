#ifndef GPU_BELLMAN_H_2
#define GPU_BELLMAN_H_2

#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <hip/hip_runtime.h>

using namespace std;

const int INF = 1e9;
const int max_concurrent = 212992;

int count_edges(const vector<vector<pair<int, int>>>& adj_list);
vector<int> ideal_dimensions(int num_ops);
void initialize_data(const vector<vector<pair<int, int>>>& adj_list, int* edges_u, int* edges_v, 
int* edges_weight, int* dist, int num_nodes);
__global__ void modify_edges(int* edges_u, int* edges_v, int* edges_weight,
int* dist, int num_edges, int rounds); 
__global__ void check_for_completion(int* edges_u, int* edges_v, int* edges_weight, 
int* modified, int* dist, int num_edges);
vector<int> gpu_bellman(const vector<vector<pair<int, int>>>& adj_list);

#endif