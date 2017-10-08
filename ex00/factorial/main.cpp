#include <iostream>
#include <chrono>
#include <cmath>


template<typename T>
T recurFact(const T n) {
    int n_int = n;
    if (n_int <= 1) {
        return 1;
    } else {
        return n_int * recurFact(n_int-1);
    }
}

template<typename T>
T iterFact(const T n) {
    T fact = 1;

    for (int i = 1; i <= n; ++i) {
        fact *= i;
    }

    return fact;
}

template <typename T>
double funcTimer(T (f)(T), int *input, int inputL) {
    auto start = std::chrono::system_clock::now();

    for (int i = 0; i < inputL; ++i) {
        f(input[i]);
    }

    auto end = std::chrono::system_clock::now();
    auto elapsed = end - start;

    return elapsed.count();
}

int* createRandNums(int N, int k1, int k2) {
    int *randNums = new int[N];
    srand(time(NULL));

    for (int i = 0; i < N; ++i) {
        randNums[i] = k1 + (std::rand() % (k2 - k1 + 1));
    }

    return randNums;
}


int main() {
    int N = std::pow(10,7), k1 = 5, k2 = 25;

    int *randNums = createRandNums(N,k1,k2);

    double recFactTime =funcTimer(recurFact<long>, randNums, N);
    double iterFactTime = funcTimer(iterFact<long>, randNums, N);

    std::cout << "Recursive: " << recFactTime	<< std::endl;
    std::cout << "Iterative: " <<  iterFactTime << std::endl;
    std::cout << "Ratio: " << recFactTime / iterFactTime << std::endl;

    return 0;
}

