#include <eigen3/Eigen/Dense>

#include <cmath>
#include <iostream>
#include <limits>


using namespace Eigen;

/* @brief Newton's method to approximate $x^{(0)}$
 * @param[in] x0_ Initial guess
 * @param[out] x0 Final estimation of $x^{(0)}$, given convergence of Newton's
 * method
 */
double newton_arctan(double x0_) {
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

    return x0;
}


int main() {
    // Netwon's method
    const double x0_ = 2; // Initial guess
    double x0 = newton_arctan(x0_);

    return 0;
}

