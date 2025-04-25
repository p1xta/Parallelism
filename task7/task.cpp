#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <memory>
#include <cstring>
#include <cublas_v2.h>
#include <nvtx3/nvToolsExt.h>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

void initialize(double* A, double* Anew, size_t size) {
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

void deallocate(double* A, double* Anew, size_t size) {
    free(A);
    free(Anew);
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
    std::cout << "Program started!\n\n";

    double* A = (double *)malloc(sizeof(double) * size * size);
    double* Anew = (double *)malloc(sizeof(double) * size * size);

    double* diff;
    int max_error_index = 0;
    cudaMalloc((void**)&diff, sizeof(double) * size * size);

    nvtxRangePushA("init");
    initialize(A, Anew, size);
    nvtxRangePop();

    auto cublas_deleter = [](cublasHandle_t* handle) {
        if (handle && *handle) {
            cublasDestroy(*handle);
            delete handle;
        }
    };
    std::unique_ptr<cublasHandle_t, decltype(cublas_deleter)> handle(new cublasHandle_t, cublas_deleter);
    cublasCreate(handle.get());

    double error = accuracy + 1.0;
    int iteration = 0;

    const auto start{std::chrono::steady_clock::now()};
    nvtxRangePushA("while");
    #pragma acc data copy(A[0:size*size], Anew[0:size*size]) create(diff[0:size*size])
    {
        while (error > accuracy && iteration < max_iterations) {
            nvtxRangePushA("calc");
    
            #pragma acc parallel loop collapse(2) present(A, Anew)
            for (int i = 1; i < size-1; ++i) {
                for (int j = 1; j < size-1; ++j) {
                    Anew[i*size + j] = 0.25 * (A[(i+1)*size + j] + A[(i-1)*size + j] + 
                                              A[i*size + j-1] + A[i*size + j+1]);
                }
            }
            if (iteration % 1000 == 0) {
                #pragma acc parallel loop collapse(2) present(A, Anew)
                for (int i = 1; i < size-1; i++) {
                    for (int j = 1; j < size-1; j++) {
                        diff[i * size + j] = fabs(A[i * size + j] - Anew[i * size + j]);
                    }
                }
                #pragma acc host_data use_device(diff) 
                {
                    cublasIdamax(*handle, size * size, diff, 1, &max_error_index);
                    cudaMemcpy(&error, &diff[max_error_index-1], sizeof(double), cudaMemcpyDeviceToHost);
                }
            }
            nvtxRangePop();
            nvtxRangePushA("swap");
            #pragma acc parallel loop collapse(2) present(A, Anew)
            for (int i = 1; i < size-1; i++) {
                for (int j = 1; j < size-1; j++) {
                    A[i * size + j] = Anew[i * size + j];
                }
            }
            nvtxRangePop();
            iteration++;
        }
        nvtxRangePop();
    }
    
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};

    std::cout << "Time:  " << elapsed_seconds.count() << " sec\n";
    std::cout << "Iterations:  " << iteration-1 << "\n";
    std::cout << "Error value:  " << error << std::endl;

    deallocate(A, Anew, size);
    cudaFree(diff);
    return 0;
}