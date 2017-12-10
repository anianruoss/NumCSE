#include "golubwelsh.hpp"
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Eigenvalues>
#include <cmath>
#include <iomanip>
#include <iostream>

#define PI M_PI
#define PI_HALF M_PI_2

//! @brief Compute $\int_{-\infty}^\infty f(x) dx$ using transformation $x = \cot(t)$
//! @tparam Function template type for function handle f (e.g.\ lambda function)
//! @param[in] n number of Gauss points
//! @param[in] f integrand
//! @return Approximation of integral $\int_{-\infty}^\infty f(x) dx$
template <class Function>
double quadinf(const int n, Function &&f) {
    Eigen::VectorXd w, x;

    // Compute nodes and weights of Gauss quadrature rule
    // using Golub-Welsh algorithm
    golubwelsh(n, w, x);

    // TODO
}


int main() {
    // Number of max Gauss pts.
    const int N = 100;

    // Integrand and exact integral
    auto f = [] (double t) {
        return std::exp(-std::pow((t-1),2));
    };

    // Exact value of integrand
    double I_ex = std::sqrt(PI);

    // NOTE: We observe exponential convergence
    int sep = 12;
    std::cout << std::setw(sep) << "Nodes"
              << std::setw(sep) << "Quadrature"
              << std::setw(sep) << "Exact"
              << std::setw(sep) << "Error"
              << std::endl;

    for (int n = 1; n <= N; ++n) {
        // Value of integrant approximated
        double QS = quadinf(n, f);

        std::cout << std::setw(sep) << n
                  << std::setw(sep) << QS
                  << std::setw(sep) << I_ex
                  << std::setw(sep) << std::abs(QS - I_ex)
                  << std::endl;
    }

    return 0;
}

