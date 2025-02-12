#include <omp.h>
#include <iostream>
#include <chrono>
#include <vector>

void matrix_mult_serial(double* matrix, double* vector, double* res, int N) {
    for (int i = 0; i < N; ++i) {
        res[i] = 0.0;
        for (int j = 0; j < N; ++j) {
            res[i] += matrix[i * N + j] * vector[j];
        }
    }
}

void matrix_mult_parallel(double* matrix, double* vector, double* res, int N, int n_threads) {
    #pragma omp parallel num_threads(n_threads) 
    {
        #pragma omp for
        for (int i = 0; i < N; ++i) {
            res[i] = 0.0;
            for (int j = 0; j < N; ++j) {
                res[i] += matrix[i * N + j] * vector[j];
            }
        }
    }
    
}

double run_parallel(int N, int num_threads) {
    double *a, *b, *c;
    // std::vector<double> a, b, c;
    a = (double*)malloc(sizeof(*a) * N * N);
    b = (double*)malloc(sizeof(*b) * N);
    c = (double*)malloc(sizeof(*c) * N);
    // a = std::vector<double>(N*N);
    // b = std::vector<double>(N);
    // c = std::vector<double>(N);
    #pragma omp parallel num_threads(40)
    {   
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                a[i * N + j] = i + j;
            c[i] = 0.0;
        }    
    }
    for (int j = 0; j < N; j++)
        b[j] = j;

    const auto start{std::chrono::steady_clock::now()};

    matrix_mult_parallel(a, b, c, N, num_threads);
    
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};
    free(a); free(b); free(c);
    return elapsed_seconds.count();
}

double run_serial(int N) {
    double *a, *b, *c;
    a = (double*)malloc(sizeof(*a) * N * N);
    b = (double*)malloc(sizeof(*b) * N);
    c = (double*)malloc(sizeof(*c) * N);
    
    #pragma omp parallel num_threads(40)
    {   
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                a[i * N + j] = i + j;
            c[i] = 0.0;
        }    
    }
    for (int j = 0; j < N; j++)
        b[j] = j;

    const auto start{std::chrono::steady_clock::now()};

    matrix_mult_serial(a, b, c, N);
    
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};
    free(a); free(b); free(c);
    return elapsed_seconds.count();
}

int main() {

    int threads[8] = {1, 2, 4, 7, 8, 16, 20, 40};

    int size = 20000;
    std::cout << "\n\n20.000x20.000 matrix\n";

    std::cout << "  Serial program\n";
    double serial_time = run_serial(size);
    std::cout << "Time: " << serial_time << "\n";

    std::cout << "  Parallel program\n";
    for (auto& n_threads : threads) {
        double parallel_time = run_parallel(size, n_threads);
        std::cout << "On " << n_threads << " threads:\n";
        std::cout << "Time: " << parallel_time << "s\n";
        std::cout << "Boost: " << serial_time / parallel_time << "times\n\n";
    }
    
    size = 40000;
    std::cout << "\n\n40.000x40.000 matrix\n";
    
    std::cout << "   program\n";
    serial_time = run_serial(size);
    std::cout << "Time: " << serial_time << "\n";

    std::cout << "  Parallel program\n";
    for (auto& n_threads : threads) {
        double parallel_time = run_parallel(size, n_threads);
        std::cout << "On " << n_threads << " threads:\n";
        std::cout << "Time: " << parallel_time << "s\n";
        std::cout << "Boost: " << serial_time / parallel_time << "times\n\n";
    }


    return 0;
}