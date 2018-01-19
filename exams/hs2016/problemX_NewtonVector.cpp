#include <cmath>
#include <iostream>
#include <utility>

#include <Eigen/Dense>

using namespace Eigen;

/*! @brief Auxiliary function for numerical integration.
 *! @tparam Function Type for function-like object f.
 *! @param[in]  f  Integrand function object.
 *! @param[in]  qr Struct composed of quadrature nodes and weights.
 *! @param[in]  x  2-dimensional vector defining the integration domain.
 *! @return  Numerical integration of $f$ on $[x_1, x_2]$.
 */
template<class Function, class QuadRule>
double integrate(const Function & f, const QuadRule & qr, const Vector2d & x) {
    VectorXd nodes = qr.nodes;
    VectorXd weights = qr.weights;
    assert(nodes.size() == weights.size() &&
           "Nodes and weights of QuadRule have different lengths");

    auto nodePullback = [&x] (double c) {
        return c * (x(1) - x(0)) + x(0);
    };

    return (x(1)-x(0))*weights.dot(nodes.unaryExpr(nodePullback).unaryExpr(f));
}

/*! @brief Approximate $x$ s.t. $F(x) = 0$
 *! Using Newton's method in higher dimensions and a Quadrature rule for integrals.
 *! @tparam     Function The type for the function-like object f.
 *! @param[in]  f        A 1-dimensional function used in the intrgrands.
 *! @param[in]  qr       Struct composed of quadrature nodes and weights.
 *! @param[in]  atol     Absolute tolerance for correction-based termination.
 *! @param[in]  rtol     Relative tolerance for correction-based termination.
 *! @param[in]  maxit    Maximum number of Newton's step.
 *! @return              Final approximation of $a$ and $b$.
 */
template<class Function, class QuadRule>
std::pair<double, double> getIntv(const Function & f, const QuadRule & qr,
                                  double atol, double rtol,
                                  unsigned int maxit=10) {
    std::pair<double, double> x_end;

    Vector2d x;
    x << 0, 1;

    Vector2d x_new = x;

    auto exp_f = [&] (double t) {
        return std::exp(f(t));
    };

    for (unsigned int itr = 0; itr < maxit; ++itr) {
        MatrixXd Dinv(2,2);
        Dinv << exp_f(x(1)) - 2.*x(1), 1. - f(x(1)),
             exp_f(x(0)) + 2.*x(0), -1. - f(x(0));
        Dinv /= (Dinv(1,1) * Dinv(0,0) - Dinv(0,1) * Dinv(1,0));

        Vector2d F;
        F << integrate(f, qr, x) - x(0) - x(1),
        integrate(exp_f, qr, x) - 1 - x(0)*x(0) - x(1)*x(1);

        x_new = x - Dinv*F;
        double r = (x - x_new).norm();

        if (r < atol || r < rtol * x_new.norm()) {
            break;
        }

        x = x_new;
    }

    x_end = {x_new(0), x_new(1)};

    return x_end;
}

//! Nodes and weights of the quadrature rule.
struct GaussRule {
    GaussRule() {
        nodes   << 0.5*(1.-std::sqrt(3./5)), 0.5, 0.5*(1.+std::sqrt(3./5));
        weights << 5./18.,4./9.,5./18.;
    }

    Vector3d nodes;
    Vector3d weights;
};


int main() {
    // Integrand.
    auto f = [&] (double t) {
        return t;
    };

    // Custom quadrature rule
    GaussRule qr;

    // Approximation of a and b.
    auto ab = getIntv(f, qr, 1e-6, 1e-6);

    // Output approximate (a,b) and reference solution.
    std::cout << "Newton's method estimate is = "
              << "(" << ab.first  << ", " << ab.second << ")" << std::endl;
    std::cout << "while the exact solution is = "
              << "(-0.696257, 1.30374)" << std::endl;

    return 0;
}

