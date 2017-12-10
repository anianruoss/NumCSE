# include "gaussquad.hpp"
# include <eigen3/Eigen/Dense>
# include <cmath>
# include <vector>


/*!
 * \brief nonSmoothIntegrand Approximte the integral $\int_{-1}^1 \arcsin(x) f(x) dx$
 *
 * \tparam Function A function object with operator()(double)
 * \param fh Will pass the integrand
 * \param I_ex Exact value of integral
 * \return Value of integral
 */
template <class Function>
void nonSmoothIntegrand(const Function& fh, const double I_ex) {

    //TODO

}

/*!
 * \brief smoothIntegrand Approximte the integral $\int_{-1}^1 \arcsin(x) f(x) dx$
 * Ensures that convergenge is expoenential using appropriate transformation,
 * provided $f$ is a smooth function.
 * \tparam Function A function object with operator()(double)
 * \param fh Will pass the integrand
 * \param I_ex Exact value of integral
 * \return Value of integral
 */
template <class Function>
void smoothIntegrand(const Function& f, const double I_ex) {

    //TODO

}


int main() {
    // "Exact" value of integral
    const double I_ex = 0.870267525725852642;

    // $f(x) = \sinh x$
    std::function<double(double)> f = [](double x) {
        return std::sinh(x);
    };

    // PART 1
    nonSmoothIntegrand(f, I_ex);

    // PART 2
    smoothIntegrand(f, I_ex);

    return 0;
}

