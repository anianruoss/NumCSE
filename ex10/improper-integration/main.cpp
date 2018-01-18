#include "golubwelsh.hpp"

#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Eigenvalues>
#include <mgl2/mgl.h>

#include <cmath>
#include <iomanip>
#include <iostream>

#define PI M_PI
#define PI_HALF M_PI_2


using namespace Eigen;

//! @brief Compute $\int_{-\infty}^\infty f(x) dx$ using transformation $x = \cot(t)$
//! @tparam Function template type for function handle f (e.g.\ lambda function)
//! @param[in] n number of Gauss points
//! @param[in] f integrand
//! @return Approximation of integral $\int_{-\infty}^\infty f(x) dx$
template <class Function>
double quadInf(const int n, Function &&f) {
    // Compute nodes and weights of Gauss quadrature using Golub-Welsh algorithm
    VectorXd weights, nodes;
    golubwelsh(n, weights, nodes);

    auto g = [f] (double x) {
        return f(std::cos(x)/std::sin(x)) / std::pow(std::sin(x), 2);
    };

    auto nodePullback = [] (double x) {
        return (1. + x) * PI_HALF;
    };

    return PI_HALF * weights.dot(nodes.unaryExpr(nodePullback).unaryExpr(g));
}


int main() {
    // Number of max Gauss pts.
    const int N = 100;
    VectorXd pts = VectorXd::LinSpaced(N, 1, N);
    VectorXd errors = VectorXd::Zero(N);

    // Integrand and exact integral
    auto f = [] (double t) {
        return std::exp(-std::pow((t-1),2));
    };

    // Exact value of integrand
    double I_ex = std::sqrt(PI);

    // NOTE: We observe exponential convergence
    int sep = 15;
    std::cout << std::setw(sep) << "Nodes"
              << std::setw(sep) << "Quadrature"
              << std::setw(sep) << "Exact"
              << std::setw(sep) << "Error"
              << std::endl;

    for (int n = 1; n <= N; ++n) {

        // Value of integrant approximated
        double QS = quadInf(n, f);
        errors(n-1) = std::abs(QS - I_ex);

        std::cout << std::setw(sep) << n
                  << std::setw(sep) << QS
                  << std::setw(sep) << I_ex
                  << std::setw(sep) << errors(n-1)
                  << std::endl;
    }

    mglData datx, daty;
    datx.Link(pts.data(), pts.size());
    daty.Link(errors.data(), errors.size());

    mglGraph gr;
    gr.Title("Improper Integration Convergence");
    gr.SetRanges(1, N, errors.minCoeff(), errors.maxCoeff());
    gr.SetFunc("x", "lg(y)");
    gr.SetTickSkip(false);
    gr.Plot(datx, daty, "b+");
    gr.AddLegend("Error", "b+");
    gr.Label('x', "# Quadrature Points", 0);
    gr.Axis();
    gr.Legend();
    gr.WriteFrame("plots/quadInf-convergence.eps");

    return 0;
}

