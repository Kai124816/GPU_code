#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <hip/hip_runtime.h>
#include "gpu_bellman.h"

using namespace std;

int count_edges(const vector<vector<pair<int, int>>>& adj_list)
{
    int edge_count = 0;
    for (int i = 0; i < adj_list.size(); ++i) 
    {
        edge_count += adj_list[i].size();
    } 
    return edge_count;
}


vector<int> ideal_dimensions(int num_threads)
{
    vector<int> x_y;

    if (num_threads > 1024)
    {
        x_y.push_back(1024);
        x_y.push_back((num_threads + 1023) / 1024);
    }
    else
    {
        int block_size = 64;
        int x_dim = ((num_threads + block_size - 1) / block_size) * block_size;
        x_y.push_back(x_dim);
        x_y.push_back(1);
    }

    return x_y;
}


void initialize_data(const vector<vector<pair<int, int>>>& adj_list, int* edges_u, int* edges_v, int* edges_weight, 
    int* active, int* dist, int num_nodes)
{
    int edge_count = 0;
    for (int i = 0; i < num_nodes; ++i) 
    {
        active[i] = 0;
        dist[i] = INF;
        for (auto [v, w] : adj_list[i]) 
        {
            edges_u[edge_count] = i;
            edges_v[edge_count] = v;
            edges_weight[edge_count] = w;
            edge_count++;
        }
    }
    active[0] = 1;
    dist[0] = 0;
}


__global__ void modify_edge(
    int* edges_u, int* edges_v, int* edges_weight,
    int* active, int* dist,
    int num_edges
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_edges || active[edges_u[idx]] == 0) return;

    int src = edges_u[idx];
    int dest = edges_v[idx];
    int weight = edges_weight[idx];

    int new_dist = dist[src] + weight;

    if (new_dist < dist[dest]) {
        atomicMin(&dist[dest], new_dist);
        atomicExch(&active[dest], 2); 
    }
}


__global__ void modify_active_edges(int* active, int* modified, int num_nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    if (active[idx] == 0 || active[idx] == 1) {
        active[idx] = 0;
    } else {
        active[idx] = 1;
        atomicAdd(modified, 1);
    }
}


vector<int> gpu_bellman(const vector<vector<pair<int, int>>>& adj_list)
{
    //Intialize vector
    vector<int> distances;

    //Determine number of edges and nodes
    int num_nodes = adj_list.size();
    int num_edges = count_edges(adj_list);
    
    //Determine Block Dimensions
    vector<int> dimensions_0 = ideal_dimensions(num_edges);
    int threads_per_block_0 = dimensions_0[0];
    int num_blocks_0 = dimensions_0[1];
    vector<int> dimensions_1 = ideal_dimensions(num_nodes);
    int threads_per_block_1 = dimensions_1[0];
    int num_blocks_1 = dimensions_1[1];

    //Initialize Data on Host
    int* h_edges_u = (int*)malloc(num_edges * sizeof(int));
    int* h_edges_v = (int*)malloc(num_edges * sizeof(int));
    int* h_edges_weight = (int*)malloc(num_edges * sizeof(int));
    int* h_active = (int*)malloc(num_nodes * sizeof(int));
    int* h_dist = (int*)malloc(num_nodes * sizeof(int));
    initialize_data(adj_list, h_edges_u, h_edges_v, h_edges_weight, h_active, h_dist, num_nodes);

    //Initialize Data on GPU
    int *d_edges_u, *d_edges_v, *d_edges_weight, *d_active, *d_dist;
    hipMalloc(&d_edges_u, sizeof(int) * num_edges);
    hipMalloc(&d_edges_v, sizeof(int) * num_edges);
    hipMalloc(&d_edges_weight, sizeof(int) * num_edges); 
    hipMalloc(&d_active, sizeof(int) * num_nodes);
    hipMalloc(&d_dist, sizeof(int) * num_nodes);

    //Copy host data to GPU
    hipMemcpy(d_edges_u, h_edges_u, sizeof(int) * num_edges, hipMemcpyHostToDevice);
    hipMemcpy(d_edges_v, h_edges_v, sizeof(int) * num_edges, hipMemcpyHostToDevice);
    hipMemcpy(d_edges_weight, h_edges_weight, sizeof(int) * num_edges, hipMemcpyHostToDevice);
    hipMemcpy(d_active, h_active, sizeof(int) * num_nodes, hipMemcpyHostToDevice);
    hipMemcpy(d_dist, h_dist, sizeof(int) * num_nodes, hipMemcpyHostToDevice);

    //Initialize modified variable on both Host and GPU
    int h_modified = 1;
    int* d_modified;
    hipMalloc(&d_modified, sizeof(int));

    //Number of rounds through bellman ford
    int rounds = 0;

    //Run Bellman Ford
    while(h_modified && rounds < num_nodes-1)
    {
        //Assign Modified to 0
        h_modified = 0;
        hipMemcpy(d_modified, &h_modified, sizeof(int), hipMemcpyHostToDevice);
        
        //Alter edges
        hipLaunchKernelGGL(modify_edge, dim3(num_blocks_0), dim3(threads_per_block_0), 0, 0, d_edges_u, d_edges_v, d_edges_weight, 
        d_active, d_dist, num_edges);

        //Reset Active Array
        hipLaunchKernelGGL(modify_active_edges, dim3(num_blocks_1), dim3(threads_per_block_1), 0, 0, d_active, d_modified, num_nodes);

        //Copy over modified data
        hipDeviceSynchronize();
        hipMemcpy(&h_modified, d_modified, sizeof(int), hipMemcpyDeviceToHost);
        rounds++;
    }

    //Negative Cycle Detection
    if(rounds == num_nodes-1)
    {
        //Assign Modified to 0
        h_modified = 0;
        hipMemcpy(d_modified, &h_modified, sizeof(int), hipMemcpyHostToDevice);
        
        //Alter edges
        hipLaunchKernelGGL(modify_edge, dim3(num_blocks_0), dim3(threads_per_block_0), 0, 0, d_edges_u, d_edges_v, d_edges_weight, 
        d_active, d_dist, num_edges);

        //Reset Active Array
        hipLaunchKernelGGL(modify_active_edges, dim3(num_blocks_1), dim3(threads_per_block_1), 0, 0, d_active, d_modified, num_nodes);

        //Copy over modified data
        hipDeviceSynchronize();
        hipMemcpy(&h_modified, d_modified, sizeof(int), hipMemcpyDeviceToHost);
        if(h_modified)
        {
            std::cerr << "Error: Negative-weight cycle detected.\n";
            return {};
        }
    }
    
    //Copy over distances to vector
    hipMemcpy(h_dist, d_dist, sizeof(int) * num_nodes, hipMemcpyDeviceToHost);
    distances.assign(h_dist, h_dist + num_nodes);

    //Free Memory
    free(h_edges_u);
    free(h_edges_v);
    free(h_edges_weight);
    free(h_active);
    free(h_dist);
    hipFree(d_edges_u);
    hipFree(d_edges_v);
    hipFree(d_edges_weight);
    hipFree(d_active);
    hipFree(d_dist);
    hipFree(d_modified);

    //Return Vector
    return distances;
}