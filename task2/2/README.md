Type "make" to compile and run
Results:

Integration f(x) on [-4, 4]  nsteps = 40000000
Result (serial): 1.772453823579; error 0.000000027326
Serial program execution time: 0.475486

Parallel program:
On 1 threads:
Time: 0.472362
Speedup: 1.00661

On 2 threads:
Time: 0.250147
Speedup: 1.90083

On 4 threads:
Time: 0.13172
Speedup: 3.60984

On 7 threads:
Time: 0.079906
Speedup: 5.95057

On 8 threads:
Time: 0.0715294
Speedup: 6.64743

On 16 threads:
Time: 0.0413707
Speedup: 11.4933

On 20 threads:
Time: 0.0357274
Speedup: 13.3087

On 40 threads:
Time: 0.0247297
Speedup: 19.2274