#include "chebPolyEval.hpp"
#include <eigen3/Eigen/Dense>
#include <mgl2/mgl.h>
#include <cmath>
#include <iostream>
#include <vector>


using namespace std;

// Compute the best approximation of the function $f$
// with Chebyshev polynomials.
// $\alpha$ is the output vector of coefficients.
template <typename Function>
void bestApproxCheb(const Function &f, Eigen::VectorXd &alpha) {
    int n=alpha.size()-1;

    //TODO
}


int main() {
    int n;

    //TODO Check the orthogonality of Chebyshev polynomials

    //TODO Test the implementation

    return 0;
}

