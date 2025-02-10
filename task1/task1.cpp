#include <iostream> 
#include <cmath>
#include <iomanip>

#define size 10000000

#ifdef DOUBLE
        using Type = double;
#else
        using Type = float;
#endif

int main() {
    Type* sin = new Type[size];
    Type sum = 0;

    for (int i = 0; i < size; i++) {
        #ifdef DOUBLE
            sin[i] = std::sin((Type)i * 2.0 * M_PI / size);
        #else 
            sin[i] = sinf((Type)i * 2.0f * M_PI / size);
        #endif
        sum += sin[i];
    }
    std::cout << std::setprecision(15);
    std::cout << std::fixed << sum << std::endl;
    
    delete[] sin;
    return 0;
}