#include <eigen3/Eigen/Dense>

#include <cmath>
#include <iostream>
#include <limits>


using namespace Eigen;

/* @brief Newton's method to approximate smallest positive intial guess
 * $x^{(0)}$ for which Newton's method does not converge for $arctan(x)$
 * @param[in] x0_ Initial guess
 * @param[out] x0 Final estimation of $x^{(0)}$, given convergence of Newton's
 * method
 */
double newtonInitialGuess(double x0_) {
    const double eps = std::numeric_limits<double>::epsilon();
    double update = 1.;
    double x0 = x0_, x1;

    auto g = [] (double x) {
        return 2.*x - (1. + x*x) * std::atan(x);
    };

    auto dg = [] (double x) {
        return 1. - 2.*x*std::atan(x);
    };

    while (std::abs(update) > eps) {
        x1 = x0 - g(x0) / dg(x0);
        update = (x1 - x0) / x1;
        x0 = x1;

        std::cout << "x0 = " << x0 << ", accuracy = " << update << std::endl;
    }
    std::cout << std::endl;

    return x0;
}

/* @brief Newton's method to approximate root of $arctan(x)$
 * @param[in] x0_ Initial guess
 * @param[out] x0 Final estimation of root, given convergence of Newton's method
 */
double newtonArctan(double x0_) {
    const size_t maxItr = 15;
    const double eps = std::numeric_limits<double>::epsilon();
    double update = 1.;
    double x0 = x0_, x1;

    for (size_t i = 0; i < maxItr; ++i) {
        if (std::abs(update) <= eps) {
            break;
        } else {
            x1 = x0 - std::atan(x0) * (1 + x0*x0);
            update = (x1 - x0) / x1;
            x0 = x1;

            std::cout << "x0 = " << x0 << ", accuracy = " << update << std::endl;

            if (i == maxItr - 1) {
                std::cerr << std::endl
                          << "Newton's method failed to converge after "
                          << maxItr << " iterations." << std::endl
						  << std::endl;
            }
        }
    }

    return x0;
}

int main() {
    std::cout << "Calculating smallest positive non-convergent initial guess"
              << std::endl;
    const double x0_ = 2; // Initial guess
    double x0 = newtonInitialGuess(x0_);

    std::cout << "Testing initial guess on Newton's method" << std::endl;
    double root = newtonArctan(x0);

    return 0;
}

