#include <iostream>


template<typename T>
T recurFact(const T n) {
    if (n <= 1) {
        return 1;
    } else {
        return n * recurFact(n-1);
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


int main() {
    std::cout << "Enter number: ";
    int n;
    std::cin >> n;

    std::cout << "Recursive factorial: " << recurFact(n) << std::endl;
    std::cout << "Iterative factorial: " << iterFact(n) << std::endl;

    return 0;
}

