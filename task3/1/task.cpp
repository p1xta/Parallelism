#include <iostream> 
#include <thread>
#include <chrono>
#include <vector> 

void matrix_mult(double* matrix, double* vector, double* res, int size, int start, int end) {
    for (int i = start; i < end; ++i) {
        res[i] = 0.0;
        for (int j = 0; j < size; ++j) {
            res[i] += matrix[i * size + j] * vector[j];
        }
    }
}

void initialize_arrays(double* matrix, double* vector, double* res, int size, int start, int end) {
    for (int i = start; i < end; i++) {
        for (int j = 0; j < size; j++)
            matrix[i * size + j] = i + j;
        vector[i] = i;
        res[i] = 0.0;
    }
}

void initialize_parallel(double* matrix, double* vector, double* res, int size, int num_threads) {
    std::vector<std::thread> threads;
    int chunk_size = size / num_threads;
    int mod = size % num_threads;

    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = start + chunk_size;
        if (i == num_threads - 1) 
            end += mod;
        
        threads.emplace_back(initialize_arrays, matrix, vector, res, size, start, end);
    }
    for (auto& t : threads)
        t.join();
}

void matrix_mult_parallel(double* matrix, double* vector, double* res, int size, int num_threads) {
    std::vector<std::thread> threads;
    int chunk_size = size / num_threads;
    int mod = size % num_threads;

    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = start + chunk_size;
        if (i == num_threads - 1) 
            end += mod;
        
        threads.emplace_back(matrix_mult, matrix, vector, res, size, start, end);
    }
    for (auto& t : threads)
        t.join();
}

double run_serial(int N) {
    double *a, *b, *c;
    a = (double*)malloc(sizeof(*a) * N * N);
    b = (double*)malloc(sizeof(*b) * N);
    c = (double*)malloc(sizeof(*c) * N);
    
    const auto start{std::chrono::steady_clock::now()};

    initialize_arrays(a, b, c, N, 0, N);
    matrix_mult(a, b, c, N, 0, N);
    
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};
    free(a); free(b); free(c);
    return elapsed_seconds.count();
}

double run_parallel(int N, int num_threads) {
    double *a, *b, *c;
    a = (double*)malloc(sizeof(*a) * N * N);
    b = (double*)malloc(sizeof(*b) * N);
    c = (double*)malloc(sizeof(*c) * N);

    const auto start{std::chrono::steady_clock::now()};

    initialize_parallel(a, b, c, N, num_threads);
    matrix_mult_parallel(a, b, c, N, num_threads);
    
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};
    free(a); free(b); free(c);
    return elapsed_seconds.count();
}

int main() {
    int size = 20000;
    int threads[8] = {1, 2, 4, 7, 8, 16, 20, 40};

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
    std::cout << "  Serial program\n";
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