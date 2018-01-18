#include <eigen3/Eigen/Dense>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

using namespace Eigen;

double norm(double x) {
    return std::abs(x);
}

double norm(const VectorXd & x) {
    return x.norm();
}

/*!
 *! \brief Implements a single step of the modified newton
 *! \tparam Scalar type of argument to function f: such as double, etc...
 *! \tparam Function type for the function f, likely a lambda function
 *! \tparam Jacobian type for the jacobian df, likely a lambda function
 *! \param[in] x previous value to use in step, also initial guess if needed
 *! \param[in] f function handle for f(x) = 0
 *! \param[in] df function handle for jacobian df of f
 *! \return next step x_{k+1} of modified Newton
 */
template <typename Scalar, class Function, class Jacobian>
Scalar mod_newt_step_scalar(const Scalar& x,
                            const Function& f,
                            const Jacobian& df) {
    //TODO
}

/**
 *! \brief Solve a scalar non-linear eq. with the modified Newton.
 * Test the empirical order of convergence of the method.
 */
void mod_newt_ord() {
    // Setting up values, functions and jacobian
    const double a = 0.123;
    //TODO
}


int main() {
    mod_newt_ord();

    return 0;
}

