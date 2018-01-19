#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

using namespace Eigen;

/*!
 *! @brief Approximation of 1 step using the implicit midpoint rule
 *! and 2 Newton's steps.
 *! @param[in]  theta    Function $\theta$.
 *! @param[in]  theta_d  The 1st derivative of $\theta$.
 *! @param[in]  h        Time step size for the implicit midpoint rule.
 *! @param[in]  y        Value of previous iteration of the implicit midpoint rule.
 *! @return              Value of the next interation of the IM rule.
 */

template<typename Functor, typename Jacobian>
Vector2d psi(Functor& theta, Jacobian& theta_d,
             double h, const Vector2d& y) {
    Vector2d z;
    // TODO: evolve $y$ using one step of implicit midpoint rule and two Newton's
    // iterations.
    return z;
}


/*!
 *! @brief Single step method for the IVP with $\theta(y) = y$.
 *! @param[in]  y0 Initial guess
 *! @param[in]  h  Time step
 *! @param[in]  N  Number of steps
 *! @param[out] yk Final estimation after $N$ steps
 */

template<typename Function>
Vector2d lfevl(const Function& theta, Vector2d y0,
               double h, unsigned int N) {
    auto f = [&theta] (const Vector2d& y) -> Vector2d {
        Vector2d y_dot;
        y_dot << -theta(y(1)), y(0);
        return y_dot;
    };
    Vector2d yk = y0;
    for(unsigned k=0; k < N; ++k) {
        Vector2d k1 = f(yk);
        Vector2d k2 = f(yk + h/2.*k1);
        Vector2d k3 = f(yk - h*k1+ 2.*h*k2);

        yk += h/6.*k1 + 2.*h/3.*k2 + h/6.*k3;
    }
    return yk;
}


int main() {
    // Initial value $y(0)$.
    Vector2d y0;
    y0 << 0., 1.;

    // Function $\theta$ and derivative
    auto theta   = [] (double xi)-> double { return std::exp(xi); };
    auto theta_d = [] (double xi)-> double { return std::exp(xi); };

    // PART 1: approximate IVP using implicit midpoint rule
    std::cout << "**** PART 1 ****" << std::endl;

    // Perform a single evolution step of size 0.1 with y(0) = [0,1]^t
    auto z = psi(theta, theta_d, 0.1, y0);
    Vector2d z_exp;
    z_exp << -0.26999952449728237269, 0.98650002377513590357;
    std::cout << "Distance between your output of 'psi' and the expected output: "
              << (z - z_exp).norm()
              << std::endl;

    // PART 2: order of convergence of RK method implemented in lfevl
    std::cout << "**** PART 2 ****" << std::endl;

    // Final time
    double T = 10.;

    // Exact value $y(10)$ at final time $T = 10$
    Vector2d y_exact;
    // Approximate solution is -2.3316..., -20.9302...;
    // Analytic solution
    double eul = std::exp(1.0);
    double sqrt2eul = std::sqrt(eul * 2.);
    double sqrteul_2 = std::sqrt(eul / 2.);
    double sech_s = 1. / std::cosh(sqrteul_2 * T);
    y_exact << - sqrt2eul * std::tanh(10 * sqrteul_2),
            1. + std::log(sech_s*sech_s);

    // TODO: empirically determine the order of convergence of "lfevl".
}
