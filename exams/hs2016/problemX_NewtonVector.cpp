#include <cmath>
#include <iostream>
#include <utility>

#include <Eigen/Dense>

using namespace Eigen;

/*!
 *! @brief Auxiliary function for numerical integration.
 *! @tparam Function Type for function-like object f.
 *! @param[in]  f  Integrand function object.
 *! @param[in]  qr Struct composed of quadrature nodes and weights.
 *! @param[in]  x  2-dimensional vector defining the integration domain.
 *! @return  Numerical integration of $f$ on $[x_1, x_2]$.
 */

template<class Function, class QuadRule>
double integrate(const Function& f, const QuadRule& qr, const Vector2d & x) {

    double I = 0;

    VectorXd   nodes = qr.nodes;
    VectorXd weights = qr.weights;
    assert(nodes.size() == weights.size() &&
           "Nodes and weights of QuadRule have different lengths");
    // TODO (optional): implement numerical integration of $f$ using QuadRule "qr".

    return I;
}


/*! @brief Approximate $x$ s.t. $F(x) = 0$
 *! Using Newton's method in higher dimensions
 *! and a Quadrature rule for integrals.
 *! @tparam     Function The type for the function-like object f.
 *! @param[in]  f        A 1-dimensional function used in the intrgrands.
 *! @param[in]  qr       Struct composed of quadrature nodes and weights.
 *! @param[in]  atol     Absolute tolerance for correction-based termination.
 *! @param[in]  rtol     Relative tolerance for correction-based termination.
 *! @param[in]  maxit    Maximum number of Newton's step.
 *! @return              Final approximation of $a$ and $b$.
 */

template<class Function, class QuadRule>
std::pair<double, double> getIntv(const Function& f, const QuadRule& qr,
                                  double atol, double rtol,
                                  unsigned maxit=10) {
    std::pair<double,double> x_end;

    Vector2d x;
    x << 0, 1;

    // TODO: implement Newton's method for the solution
    // of the multidimensional nonlinear system for problem 0.2.

    return x_end;
}


//! Nodes and weights of the quadrature rule.
struct GaussRule {
    GaussRule() {
        nodes   << 0.5*(1.-std::sqrt(3./5)),
                0.5,
                0.5*(1.+std::sqrt(3./5));
        weights << 5./18.,4./9.,5./18.;
    }

    Vector3d nodes;
    Vector3d weights;
};

int main() {
    // Testing unit.

    // Integrand.
    auto f = [&] (double t) {
        return t;
    };

    // Custom quadrature rule
    GaussRule qr;

    // Approximation of a and b.
    auto ab = getIntv(f, qr, 1e-6, 1e-6);

    // Output approximate (a,b) and reference solution.
    std::cout << "Newton's method estimate is =" << std::endl;
    std::cout << "(" << ab.first  << ", ";
    std::cout << ab.second << ")," << std::endl;
    std::cout << "while the exact solution is =" << std::endl;
    std::cout << "(-0.696257, 1.30374)" << std::endl;
}
