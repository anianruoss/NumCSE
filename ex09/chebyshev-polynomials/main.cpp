#include "chebPolyEval.hpp"
#include <eigen3/Eigen/Dense>
#include <mgl2/mgl.h>
#include <cmath>
#include <iostream>
#include <vector>


using namespace Eigen;

// Compute the best approximation of the function $f$
// with Chebyshev polynomials.
// $\alpha$ is the output vector of coefficients.
template <typename Function>
void bestApproxCheb(const Function &f, VectorXd &alpha) {
    int n = alpha.size()-1;

    //TODO
}


int main() {
    int n = 1000;

    // Check the orthogonality of Chebyshev polynomials
    std::cout << "Checking orthogonality of Chebyshev polynomials" << std::endl;

    auto roots = [n] (double x) {
        return std::cos(M_PI*(2*x+1)/(2*(n+1)));
    };

    VectorXd chebyRoots = VectorXd::LinSpaced(n+1, 0, n).unaryExpr(roots);
    MatrixXd scal = MatrixXd::Zero(n+1,n+1);

    for (int i = 0; i < n+1; ++i) {
        std::vector<double> chebyPol = chebPolyEval(n, chebyRoots(i));

        for (int k = 0; k < n+1; ++k) {
            scal(i,k) = chebyPol[k];
        }
    }

    double epsilon = std::numeric_limits<double>::epsilon();

    for (int k = 0; k < n+1; ++k) {
        for (int l = k; l < n+1; ++l) {
            double innerProd = scal.col(k).dot(scal.col(l));

            if (l == k) {
                assert(innerProd != 0);
            } else	{
                assert(innerProd < epsilon * n*n);
            }
        }
    }

    std::cout << "All checks have passed" << std::endl;


    //TODO Test the implementation

    return 0;
}

