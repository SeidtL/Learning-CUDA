#include <chrono>
#include <vector>
#include <iostream>

using std::chrono::high_resolution_clock;
using namespace std::chrono_literals;

template <typename T>
T trace(const std::vector<T>& mat, size_t rows, size_t cols);

int main() {

    std::vector<int> x(5000 * 5000);
    
    auto start = high_resolution_clock::now();
    trace(x, 5000, 5000);
    auto end = high_resolution_clock::now();
    std::cout << (end - start).count() << std::endl;
}
