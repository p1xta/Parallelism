#include <iostream> 
#include <cmath>
#include <iomanip>

#define size 10000000

#ifdef USE_DOUBLE
        using ArrayType = double;
#else
        using ArrayType = float;
#endif


int main() {
    ArrayType* sin = new ArrayType[size];
    ArrayType sum = 0;

    for (int i = 0; i < size; i++) {
        #ifdef USE_DOUBLE
            sin[i] = std::sin((ArrayType)i * 2.0 * M_PI / size);
        #else 
            sin[i] = sinf((ArrayType)i * 2.0 * M_PI / size);
        #endif
        sum += sin[i];
    }
    std::cout << std::setprecision(15);
    std::cout << std::fixed << sum << std::endl;
    
    delete[] sin;
    return 0;
}