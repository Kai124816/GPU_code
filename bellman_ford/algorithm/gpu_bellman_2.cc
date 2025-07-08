#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <hip/hip_runtime.h>
#include "gpu_bellman_2.h"

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


vector<int> ideal_dimensions(int num_ops)
{
    vector<int> x_y;

    if (num_ops >= max_concurrent)
    {
        x_y.push_back(1024);
        x_y.push_back(208);
    }
    else if (num_ops > 1024)
    {
        x_y.push_back(1024);
        x_y.push_back((num_ops + 1023) / 1024);
    }
    else
    {
        int block_size = 64;
        int x_dim = ((num_ops + block_size - 1) / block_size) * block_size;
        x_y.push_back(x_dim);
        x_y.push_back(1);
    }

    return x_y;
}


int predicted_iterations(int num_nodes, int num_edges)
{
    int ratio = num_edges/num_nodes;

    if(ratio < 2)
    {
        return num_nodes/3;
    }
    else if(ratio < 3)
    {
        return 15
    }
    else if(ratio < 4)
    {
        return 10
    }
    else
    {
        return 5
    }
}


void initialize_data(const vector<vector<pair<int, int>>>& adj_list, int* edges_u, 
    int* edges_v, int* edges_weight,  int* dist, int num_nodes)
{
    int edge_count = 0;
    for (int i = 0; i < num_nodes; ++i) 
    {
        dist[i] = INF;
        for (auto [v, w] : adj_list[i]) 
        {
            edges_u[edge_count] = i;
            edges_v[edge_count] = v;
            edges_weight[edge_count] = w;
            edge_count++;
        }
    }
    dist[0] = 0;
}


__global__ void modify_edges(
    int* edges_u, int* edges_v, int* edges_weight,
    int* dist, int num_edges, int rounds
) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int global_size = blockDim.x * gridDim.x;

    for(int i = 0; i < rounds; i++)
    {
        for (int j = idx; j < num_edges; j += global_size) 
        {
            int src = edges_u[j];
            int dest = edges_v[j];
            int weight = edges_weight[j];
            int new_dist = dist[src] + weight;

            if (new_dist < dist[dest]) {
                atomicMin(&dist[dest], new_dist);
            }
        }
    }
}


__global__ void check_for_completion(
    int* edges_u, int* edges_v, int* edges_weight,
    int* modified, int* dist, int num_edges
) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int global_size = blockDim.x * gridDim.x;

    
    for (int i = idx; i < num_edges; i += global_size) 
    {
        int src = edges_u[i];
        int dest = edges_v[i];
        int weight = edges_weight[i];
        int new_dist = dist[src] + weight;

        if (new_dist < dist[dest]) {
            atomicMin(&dist[dest], new_dist);
            atomicAdd(modified, 1);
        }
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
    vector<int> dimensions = ideal_dimensions(num_edges);
    int threads_per_block = dimensions[0];
    int num_blocks = dimensions[1];

    //Initialize Data on Host
    int* h_edges_u = (int*)malloc(num_edges * sizeof(int));
    int* h_edges_v = (int*)malloc(num_edges * sizeof(int));
    int* h_edges_weight = (int*)malloc(num_edges * sizeof(int));
    int* h_dist = (int*)malloc(num_nodes * sizeof(int));
    initialize_data(adj_list, h_edges_u, h_edges_v, h_edges_weight, h_dist, num_nodes);

    //Initialize Data on GPU
    int *d_edges_u, *d_edges_v, *d_edges_weight, *d_dist;
    hipMalloc(&d_edges_u, sizeof(int) * num_edges);
    hipMalloc(&d_edges_v, sizeof(int) * num_edges);
    hipMalloc(&d_edges_weight, sizeof(int) * num_edges); 
    hipMalloc(&d_dist, sizeof(int) * num_nodes);

    //Copy host data to GPU
    hipMemcpy(d_edges_u, h_edges_u, sizeof(int) * num_edges, hipMemcpyHostToDevice);
    hipMemcpy(d_edges_v, h_edges_v, sizeof(int) * num_edges, hipMemcpyHostToDevice);
    hipMemcpy(d_edges_weight, h_edges_weight, sizeof(int) * num_edges, hipMemcpyHostToDevice);
    hipMemcpy(d_dist, h_dist, sizeof(int) * num_nodes, hipMemcpyHostToDevice);

    //Initialize modified variable on both Host and GPU
    int h_modified = 1;
    int* d_modified;
    hipMalloc(&d_modified, sizeof(int));

    //Number of rounds through bellman ford
    int total_rounds = 0;
    int rounds_per_iteration = predicted_iterations(num_nodes, num_edges);

    //Run Bellman Ford
    while(h_modified && total_rounds < num_nodes - 1)
    {
        //Assign Modified to 0
        h_modified = 0;
        hipMemcpy(d_modified, &h_modified, sizeof(int), hipMemcpyHostToDevice);
        
        //Alter edges
        hipLaunchKernelGGL(modify_edges, dim3(num_blocks), dim3(threads_per_block), 
        0, 0, d_edges_u, d_edges_v, d_edges_weight, d_dist, num_edges, rounds_per_iteration - 1);

        //Check for completion
        hipLaunchKernelGGL(check_for_completion, dim3(num_blocks), dim3(threads_per_block), 
        0, 0, d_edges_u, d_edges_v, d_edges_weight, d_modified, d_dist, num_edges);

        //Copy over modified data
        hipDeviceSynchronize();
        hipMemcpy(&h_modified, d_modified, sizeof(int), hipMemcpyDeviceToHost);
        total_rounds += rounds_per_iteration;

        if(total_rounds + rounds_per_iteration > num_nodes - 1){
            rounds_per_iteration = num_nodes - 1 - total_rounds;
        }
    }

    //Negative Cycle Detection
    if(rounds >= num_nodes-1)
    {
        //Assign Modified to 0
        h_modified = 0;
        hipMemcpy(d_modified, &h_modified, sizeof(int), hipMemcpyHostToDevice);
        
        //Check for completion
        hipLaunchKernelGGL(check_for_completion, dim3(num_blocks), dim3(threads_per_block), 
        0, 0, d_edges_u, d_edges_v, d_edges_weight, d_modified, d_dist, num_edges);

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
    free(h_dist);
    hipFree(d_edges_u);
    hipFree(d_edges_v);
    hipFree(d_edges_weight);
    hipFree(d_dist);
    hipFree(d_modified);

    //Return Vector
    return distances;
}