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
Scalar mod_newt_step_scalar(const Scalar & x, const Function & f,
                            const Jacobian & df) {
    Scalar y = x + f(x) / df(x);
    return y - f(y) / df(x);
}

/*!
 *! \brief Performs many steps of an iteration and terminates when convergence
 *! or maximum number of iterations has been reached.
 *! \tparam StepFunction type for the step function handle
 *! \tparam Vector argument type passed to the iteration function
 *! \tparam ErrorFunction type for the error function computing error of the
 *! method
 *! \param[in] step Function implementing the step x_{k+1} = step(x_{k}),
 *! signature Vector(const Vector&)
 *! \param[in,out] x initial data (as input) and final iteration (as output)
 *! \param[in] errf function implementing the norm of the error (errf(x)) for
 *! termination condition
 *! \param[in] eps tolerance to break iterations when res(x) < eps
 *! \param[in] max_itr maximal number of iterations
 */
template <class StepFunction, class Vector, class ErrorFunction>
bool nonlinear_solver(const StepFunction & step, Vector & x,
                      const ErrorFunction & errf,
                      const int max_itr = 100, const double eps = 1e-8) {
    Vector x_new = x;
    double r = 1;

    for (int itr = 0; itr < max_itr; ++itr) {
        r = errf(x);
        std::cout<<"[Step " <<itr<< "] Error: "<< r <<std::endl;

        // Advance to next step, $x_{new}$ becomes $x_{k+1}$
        x_new = step(x);

        if (r < eps * norm(x)) {
            std::cout << "[CONVERGED] in " << itr << " iterations"
                      << std::endl << std::endl;

            return true;
        }

        x = x_new;
    }

    std::cout << "[DID NOT CONVERGE] reached MAX iteration = " << max_itr
              << std::endl << std::endl;

    return false;
}

/**
 *! \brief Solve a scalar non-linear eq. with the modified Newton.
 * Test the empirical order of convergence of the method.
 */
void mod_newt_ord() {
    const double a = 0.123;
    const double exactSol = std::tan(a);

    auto f = [a] (double x) {
        return std::atan(x) - a;
    };

    auto df = [] (double x) {
        return 1. / (1. + x*x);
    };

    std::vector<double> solutions;
    std::vector<double> errors;

    auto errf = [&errors, exactSol] (double & x) {
        double e = std::abs(x - exactSol);
        errors.push_back(e);
        return e;
    };

    double x_scalar = 5.;
    solutions.push_back(x_scalar);

    auto newt_scalar_step = [&solutions, &f, &df] (double x) -> double {
        double x_new = mod_newt_step_scalar(x, f, df);
        solutions.push_back(x_new);
        return x_new;
    };

    nonlinear_solver(newt_scalar_step, x_scalar, errf);
    solutions.pop_back();

    auto space = std::setw(15);
    std::cout << space << "approximation"
              << space << "error"
              << space << "order"
              << std::endl;

    for (size_t	i = 0; i < solutions.size(); ++i) {
        std::cout << space << solutions.at(i)
                  << space << errors.at(i);

        if (i > 1) {
            double diff1 = std::log(errors.at(i)) - std::log(errors.at(i-1));
            double diff2 = std::log(errors.at(i-1)) - std::log(errors.at(i-2));
            std::cout << space << diff1 / diff2;
        }
        std::cout << std::endl;
    }
}


int main() {
    mod_newt_ord();

    return 0;
}

