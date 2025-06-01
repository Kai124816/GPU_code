#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <hip/hip_runtime.h>


__global__ void updateSum(float* running_sum, float* data, int vector_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vector_size) {
        atomicAdd(running_sum, data[idx]);
    }
}


__global__ void updateVariance(float* running_variance, float* data, float* mean, int vector_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vector_size) {
        float mean_val = *mean;
        float diff = data[idx] - mean_val;
        atomicAdd(running_variance, diff * diff);
    }
}


// Function to read floats from a CSV file into a vector
std::vector<float> readCSVFloats(const std::string& filename) {
    std::vector<float> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            try {
                float value = std::stof(cell);
                data.push_back(value);
            }
        }
    }

    file.close();
    return data;
}


// Main program
int main() {
    std::string filename = "random_numbers.csv";  // Replace with your file name

    std::vector<float> values = readCSVFloats(filename);

    size_t array_len = values.size();    //Define the number of values

    std::cout << "Read " << array_len << " float values from " << filename << std::endl;

    float* d_values;     //Array that is copied onto GPU
    size_t numElements = values.size();
    size_t sizeInBytes = numElements * sizeof(float);

    // Allocate device memory for data array
    hipMalloc(&d_values, sizeInBytes);

    // Copy data from host vector to device data array
    hipMemcpy(d_values, values.data(), sizeInBytes, hipMemcpyHostToDevice);

    float h_sum = 0.0f;           // Host Running sum
    float* d_sum;                 // Device Running sum

    hipMalloc(&d_sum, sizeof(float));                        // Allocate device memory
    hipMemcpy(d_sum, &h_sum, sizeof(float), hipMemcpyHostToDevice);  //Copy host running sum to device running sum

    int threadsPerBlock = 256;                      // Define the number of threads per block
    int numBlocks = (array_len + threadsPerBlock - 1) / threadsPerBlock;  //Define number of blocks

    hipLaunchKernelGGL(updateSum, dim3(numBlocks), dim3(threadsPerBlock), 0, 0, d_sum, d_values, array_len); //Launch Update sum function

    hipMemcpy(&h_sum, d_sum, sizeof(float), hipMemcpyDeviceToHost); //Copy the device running sum to the host running sum

    float h_mean = h_sum/array_len;   //Calculate mean

    std::cout << "Mean: " << std::fixed << std::setprecision(2) << h_mean << std::endl;  //Print mean

    float* d_mean;   // Mean for device      

    hipMalloc(&d_mean, sizeof(float));                        // Allocate device memory
    hipMemcpy(d_mean, &h_mean, sizeof(float), hipMemcpyHostToDevice);  //Copy host mean to device mean

    h_sum = 0.0f;   //Update running sum for variance
    hipMemcpy(d_sum, &h_sum, sizeof(float), hipMemcpyHostToDevice);  //Update running sum on device

    hipLaunchKernelGGL(updateVariance, dim3(numBlocks), dim3(threadsPerBlock), 0, 0, d_sum, d_values, d_mean, array_len);

    hipMemcpy(&h_sum, d_sum, sizeof(float), hipMemcpyDeviceToHost); //Copy the device running sum to the host running sum

    float variance = h_sum/array_len;

    std::cout << "Standard Deviation: " << std::fixed << std::setprecision(2) << sqrt(variance) << std::endl;

    hipFree(d_sum);
    hipFree(d_values);
    hipFree(d_mean);

    return 0;
}




