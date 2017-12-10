#include "gaussquad.hpp"
#include <eigen3/Eigen/Dense>
#include <mgl2/mgl.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>


using namespace Eigen;

/*!
 * \brief nonSmoothIntegrand Approximte the integral $\int_{-1}^1 \arcsin(x) f(x) dx$
 *
 * \tparam Function A function object with operator()(double)
 * \param fh Will pass the integrand
 * \param I_ex Exact value of integral
 * \return Value of integral
 */
template <class Function>
void nonSmoothIntegrand(const Function &fh, const double I_ex) {
    int N = 50;
    int sep = 15;

    std::cout << "--> Approximating Integral of Non-Smooth Function"
              << std::endl
              << std::setw(5) << "Nodes"
              << std::setw(sep) << "Quadrature"
              << std::setw(sep) << "Exact"
              << std::setw(sep) << "Error"
              << std::endl;

    auto f = [fh] (double t) {
        return std::asin(t)*fh(t);
    };

    QuadRule qr;
    VectorXd errors = VectorXd::Zero(N);

    for (int n = 1; n <= N; ++n) {
        gaussquad(n, qr);

        double QS = qr.weights.dot(qr.nodes.unaryExpr(f));
        errors(n-1) = std::abs(QS - I_ex);

        std::cout << std::setw(5) << n
                  << std::setw(sep) << QS
                  << std::setw(sep) << I_ex
                  << std::setw(sep) << errors(n-1)
                  << std::endl;
    }

    VectorXd refx = VectorXd::LinSpaced(N, 1, N);

    mglData datx, daty;
    datx.Link(refx.data(), refx.size());
    daty.Link(errors.data(), refx.size());

    double yMax = errors.maxCoeff();
    double yMin = errors.minCoeff();

    mglGraph gr;
    gr.Title("Maximum Error");
    gr.SetRanges(1, N, yMin, yMax);
    gr.SetFunc("x","lg(y)");
    gr.Axis();
    gr.Plot(datx, daty, "b+");
    gr.AddLegend("f(t) = sinh(t)","b+");
    gr.Label('x',"Step [N]", 0);
    gr.Label('y', "Error", 0);
    gr.Legend();
    gr.WriteFrame("plots/gaussQuadrature-nonSmoothError.eps");
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

    nonSmoothIntegrand(f, I_ex);

    smoothIntegrand(f, I_ex);

    return 0;
}

