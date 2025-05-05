#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <boost/program_options.hpp>
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvtx3/nvToolsExt.h>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

__global__ void grid_kernel(double* A, double* Anew, int size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && i < size-1 && j > 0 && j < size-1) {
        Anew[i*size + j] = 0.25 * (A[(i+1)*size + j] + A[(i-1)*size + j] +
                                   A[i*size + j+1] + A[i*size + j-1]);
    }
}

__global__ void error_kernel(double* A, double* Anew, double* block_max_errors, int size) {
    using BlockReduce = cub::BlockReduce<double, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double local_max = 0.0;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int idx = thread_id; idx < size * size; idx += total_threads) {
        int i = idx / size;
        int j = idx % size;
        if (i > 0 && i < size-1 && j > 0 && j < size-1) {
            double diff = fabs(A[idx] - Anew[idx]);
            if (diff > local_max) 
                local_max = diff;
        }
    }
    double block_max = BlockReduce(temp_storage).Reduce(local_max, cub::Max());
    if (threadIdx.x == 0) 
        block_max_errors[blockIdx.x] = block_max;
}

__global__ void copy_kernel(double* A, const double* Anew, int size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 1 && i < size - 1 && j >= 1 && j < size - 1) {
        A[i * size + j] = Anew[i * size + j];
    }
}

void initialize_host(double* A, double* Anew, size_t size) {
    memset(A, 0, size * size * sizeof(double));
    memset(Anew, 0, size * size * sizeof(double));
    A[0] = 10.0;
    A[size-1] = 20.0;
    A[size*(size-1)] = 30.0;
    A[size*size-1] = 20.0;
    
    // border interpolation
    double top_left = A[0];
    double top_right = A[size-1];
    double bottom_left = A[size*(size-1)];
    double bottom_right = A[size*size-1];

    for (int i = 1; i < size-1; ++i) {
        A[i] = top_left + (top_right-top_left)*i / static_cast<double>(size-1); // top  
        A[size*(size-1) + i] = bottom_left + (bottom_right-bottom_left)*i / static_cast<double>(size-1); // bottom  

        A[size*i] = top_left + (bottom_left-top_left)*i / static_cast<double>(size-1); // left 
        A[size*i + size-1] = top_right + (bottom_right-top_right)*i / static_cast<double>(size-1); // right 
    }
}


int main(int argc, char* argv[]) {
    int size;
    double accuracy;
    int max_iterations;

    po::options_description desc("Options");
    desc.add_options()
        ("help", "show option description")
        ("size", po::value<int>(&size)->default_value(256), "grid size (NxN)")
        ("accuracy", po::value<double>(&accuracy)->default_value(1e-6), "maximum allowed error")
        ("max_iterations", po::value<int>(&max_iterations)->default_value(1e+6), "maximum allowed iterations");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    double error = accuracy + 1.0;
    int iteration = 0;

    size_t grid_size = size * size * sizeof(double);
    double* host_A = (double*)malloc(grid_size);
    double* host_Anew = (double*)malloc(grid_size);

    initialize_host(host_A, host_Anew, size);

    double* device_A, *device_Anew;
    cudaMalloc(&device_A, grid_size);
    cudaMalloc(&device_Anew, grid_size);
    cudaMemcpy(device_A, host_A, grid_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_Anew, host_Anew, grid_size, cudaMemcpyHostToDevice);

    int threads = 16;
    dim3 block(threads, threads);
    dim3 grid((size + threads - 1) / threads, (size + threads - 1) / threads);

    int num_blocks = 1024;
    double* d_block_max;
    cudaMalloc(&d_block_max, sizeof(double) * num_blocks);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < 100; i++) {
        grid_kernel<<<grid, block, 0, stream>>>(device_A, device_Anew, size);
        copy_kernel<<<grid, block, 0, stream>>>(device_A, device_Anew, size);
    }
    grid_kernel<<<grid, block, 0, stream>>>(device_A, device_Anew, size);
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

    std::cout << "Program started!\n\n";

    const auto start{std::chrono::steady_clock::now()};

    while (error > accuracy && iteration < max_iterations) {
        cudaGraphLaunch(instance, stream);
        cudaStreamSynchronize(stream);

        error_kernel<<<num_blocks, 256, 0, stream>>>(device_A, device_Anew, d_block_max, size);
        double* h_block_max = new double[num_blocks];
        cudaMemcpy(h_block_max, d_block_max, sizeof(double) * num_blocks, cudaMemcpyDeviceToHost);

        error = 0.0;
        for (int i = 0; i < num_blocks; ++i) 
            if (h_block_max[i] > error)
                error = h_block_max[i];

        delete[] h_block_max;
        iteration+=100;
    }

    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};

    std::cout << "Time:  " << elapsed_seconds.count() << " sec\n";
    std::cout << "Iterations:  " << iteration-1 << "\n";
    std::cout << "Error value:  " << error << std::endl;

    cudaFree(device_A);
    cudaFree(device_Anew);
    cudaFree(d_block_max);
    free(host_A);
    free(host_Anew);
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(instance);
    cudaStreamDestroy(stream);
    return 0;
}