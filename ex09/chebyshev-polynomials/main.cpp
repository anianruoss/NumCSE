#include "chebPolyEval.hpp"

#include <eigen3/Eigen/Dense>
#include <mgl2/mgl.h>

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>


using namespace Eigen;

double recclenshaw(const VectorXd &a, const double x) {
    const VectorXd::Index n = a.size() - 1;

    if (n == 0) {
        return a(0);
    } else if (n == 1) {
        return (x*a(1) + a(0));
    } else {
        VectorXd new_a(n);
        new_a << a.head(n-2), a(n-2) - a(n), a(n-1) + 2*x*a(n);

        return recclenshaw(new_a, x);
    }
}

// Compute the best approximation of the function $f$
// with Chebyshev polynomials.
// $\alpha$ is the output vector of coefficients.
template <typename Function>
void bestApproxCheb(const Function &f, VectorXd &alpha) {
    const int n = alpha.size()-1;

    auto roots = [n] (double x) {
        return std::cos(M_PI * (2.*x + 1) / (2.*(n+1)));
    };

    VectorXd chebyRoots = VectorXd::LinSpaced(n+1, 0, n).unaryExpr(roots);
    VectorXd fRoots = chebyRoots.unaryExpr(f);
    MatrixXd scal = MatrixXd::Zero(n+1,n+1);

    for (int i = 0; i < n+1; ++i) {
        std::vector<double> chebyPol = chebPolyEval(n, chebyRoots(i));
        scal.row(i) = Map<VectorXd>(chebyPol.data(), chebyPol.size());
    }

    alpha = (scal.transpose() * fRoots) / static_cast<double>(n+1);
    alpha.tail(n) *= 2.;
}


int main() {
    {
        int n = 1000;

        auto roots = [n] (double x) {
            return std::cos(M_PI*(2*x+1)/(2*(n+1)));
        };

        VectorXd chebyRoots = VectorXd::LinSpaced(n+1, 0, n).unaryExpr(roots);
        MatrixXd scal = MatrixXd::Zero(n+1,n+1);

        for (int i = 0; i < n+1; ++i) {
            std::vector<double> chebyPol = chebPolyEval(n, chebyRoots(i));
            scal.row(i) = Map<VectorXd>(chebyPol.data(), chebyPol.size());
        }

        double maxError = 0;

        for (int k = 0; k < n+1; ++k) {
            for (int l = k+1; l < n+1; ++l) {
                maxError = std::max(maxError, scal.col(k).dot(scal.col(l)));
            }

            assert(scal.col(k).dot(scal.col(k)) != 0);
        }

        std::cout << "Maximum orthogonality error: " << maxError << std::endl;
    }

    {
        int n = 20;
        int sampling_points = std::pow(10,6);

        auto f = [] (double x) {
            return 1./(std::pow(5*x,2) + 1);
        };

        VectorXd evalT = VectorXd::LinSpaced(sampling_points, -1, 1);
        VectorXd fvalT = evalT.unaryExpr(f);

        VectorXd alpha = VectorXd::Random(n+1);
        bestApproxCheb(f, alpha);

        auto tmp = std::bind(recclenshaw, alpha, std::placeholders::_1);
        VectorXd approxT = evalT.unaryExpr(tmp);

        std::cout << "Interpolation error: "
                  << (fvalT - approxT).cwiseAbs().maxCoeff()
                  << std::endl;

        mglData refx, refy;
        refx.Link(evalT.data(), evalT.size());
        refy.Link(fvalT.data(), evalT.size());

        mglData datx, daty;
        datx.Link(evalT.data(), evalT.size());
        daty.Link(approxT.data(), approxT.size());

        mglGraph gr;
        gr.SetRanges(-1, 1, 0, 1);
        gr.Axis();
        gr.Plot(refx, refy, "g");
        gr.AddLegend("Original Function", "g");
        gr.Plot(datx, daty, "b");
        gr.AddLegend("Interpolation", "b");
        gr.Legend(2);
        gr.Title("Chebyshev Polynomial Interpolation");
        gr.WriteFrame("plots/chebyPolyIpol.png");
    }

    return 0;
}

