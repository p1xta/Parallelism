#include <iostream>
#include <omp.h>
#include <math.h>
#include <chrono> 

const double PI = 3.14159265358979323846;
const double a = -4.0;
const double b = 4.0;
const int nsteps = 40000000;

double func(double x) {
    return exp(-x * x);
}

double integrate(double (*func)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += func(a + h * (i + 0.5));
    sum *= h;
    return sum;
}

double integrate_omp(double (*func)(double), double a, double b, int n, int n_threads) {
    double h = (b - a) / n;
    double sum = 0.0;
    #pragma omp parallel num_threads(n_threads)
    {
        double f = 0.0;
        #pragma omp for
        for (int i = 0; i < n; i++)
            f += func(a + h * (i + 0.5));
            #pragma omp atomic
            sum += f;
    }
    sum *= h;
    return sum;
}

double run_serial() {
    const auto start{std::chrono::steady_clock::now()};
    double res = integrate(func, a, b, nsteps);
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};
    printf("Result (serial): %.12f; error %.12f\n", res, abs(res - sqrt(PI)));
    return elapsed_seconds.count();
}

double run_parallel(int num_threads) {
    const auto start{std::chrono::steady_clock::now()};

    double res = integrate_omp(func, a, b, nsteps, num_threads);
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};
    return elapsed_seconds.count();
}

int main() {
    int threads[8] = {1, 2, 4, 7, 8, 16, 20, 40};

    std:: cout << "Integration f(x) on [" << a << ", " << b << "]  nsteps = " << nsteps << '\n';
    double tserial = run_serial();
    std::cout << "Serial program execution time: " <<  tserial << "\n\n";
    std::cout << "Parallel program:\n";

    for (auto& thread_num : threads) {
        double tparallel = run_parallel(thread_num);
        std::cout << "On " << thread_num << " threads:\n";
        std::cout << "Time: " << tparallel << '\n';
        std::cout << "Speedup: " <<  tserial / tparallel << "\n\n";
    }
    
    return 0;
}