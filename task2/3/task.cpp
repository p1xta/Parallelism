#include <iostream>
#include <omp.h>
#include <math.h>
#include <chrono> 

double epsilon = 0.000001;
double teta = 0.00001;

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

double vector_magnitude(double* vector, int len) {
    double magnitude = 0.0;
    for (int i = 0; i < len; ++i) {
        magnitude += vector[i] * vector[i];
    }
    return sqrt(magnitude);
}


double iteration_method_serial(double* A, double* b, double* x, int N) {
    double* Ax = (double*)malloc(sizeof(double) * N);
    double* x_new = (double*)malloc(sizeof(double) * N);
    double criterion = 1;

    const auto start{std::chrono::steady_clock::now()};
    while (criterion >= epsilon) {
        matrix_mult_serial(A, x, Ax, N);
        double x_new_magnitude = 0.0;
        for (int i = 0; i < N; ++i) {
            x_new[i] = Ax[i] - b[i];
            x[i] -= x_new[i] * teta;
            x_new_magnitude += x_new[i] * x_new[i];
        }
        criterion = x_new_magnitude / vector_magnitude(b, N);
    }
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};
    free(x_new); free(Ax);
    return elapsed_seconds.count();
}

double iteration_method_parallel_1_var(double* A, double* b, double* x, int N, int n_threads) {
    double* Ax = (double*)malloc(sizeof(double) * N);
    double* x_new = (double*)malloc(sizeof(double) * N);
    double criterion = 1;

    const auto start{std::chrono::steady_clock::now()};
    while (criterion >= epsilon) {
        matrix_mult_parallel(A, x, Ax, N, n_threads);
        double x_new_magnitude = 0.0;
        #pragma omp for
        for (int i = 0; i < N; ++i) {
            x_new[i] = Ax[i] - b[i];
            x[i] -= x_new[i] * teta;
            x_new_magnitude += x_new[i] * x_new[i];
        }
        criterion = x_new_magnitude / vector_magnitude(b, N);
    }
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};
    free(x_new); free(Ax);
    return elapsed_seconds.count();
}

double iteration_method_parallel_2_var(double* A, double* b, double* x, int N, int n_threads) {
    double* Ax = (double*)malloc(sizeof(double) * N);
    double* x_new = (double*)malloc(sizeof(double) * N);
    double b_length = vector_magnitude(b, N);
    double criterion = 1.0;

    const auto start{std::chrono::steady_clock::now()};
    #pragma omp parallel num_threads(n_threads) shared(criterion)
    {
        while (criterion >= epsilon) {
            double x_new_magnitude = 0.0; 
            
            #pragma omp for schedule(static)
            for (int i = 0; i < N; ++i) {
                Ax[i] = 0.0;
                for (int j = 0; j < N; ++j) {
                    Ax[i] += A[i * N + j] * x[j];
                }
            }
            #pragma omp barrier
            #pragma omp for 
            for (int i = 0; i < N; ++i) {
                x_new[i] = Ax[i] - b[i];
                x[i] -= x_new[i] * teta;
                #pragma omp atomic
                x_new_magnitude += x_new[i] * x_new[i];
            }
            #pragma omp single
            {
                criterion = x_new_magnitude / b_length;
            }
        }        
    }

    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};
    free(x_new); free(Ax);
    return elapsed_seconds.count();
}

int main() {
    int N = 10000;
    int threads[8] = {1, 2, 4, 7, 8, 16, 20, 40};

    double* A = (double*)malloc(sizeof(double) * N * N);
    double* b = (double*)malloc(sizeof(double) * N);
    double* x = (double*)malloc(sizeof(double) * N);

    for (int i = 0; i < N; ++i){
        for (int j = 0; j < N; ++j) {
            if (i == j) A[i * N + j] = 2.0;
            else A[i * N + j] = 1.0;
        }
        b[i] = static_cast<double>(N+1);
        x[i] = 0.0;
    }


    std::cout << "Serial program execution time:  " << iteration_method_serial(A, b, x, N) << " s\n";
    for (int i = 0; i < N; ++i) 
        x[i] = 0.0;
    
    std::cout << "\n\n";
    std::cout << "Parallel program 1:\n";
    for (auto& num_threads : threads) {
        std::cout << "Execution time on " << num_threads << " threads:" << iteration_method_parallel_1_var(A, b, x, N, num_threads) << " s\n";
        for (int i = 0; i < N; ++i) 
            x[i] = 0.0;
    }
    
    std::cout << "\n\n";
    std::cout << "Parallel program 2:\n";
    for (auto& num_threads : threads) {
        std::cout << "Execution time on " << num_threads << " threads:" << iteration_method_parallel_2_var(A, b, x, N, num_threads) << " s\n";
        for (int i = 0; i < N; ++i) 
            x[i] = 0.0;
    }

    std::cout << "\n\n";
    return 0;
}