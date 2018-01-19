#include "chebPolyEval.hpp"

#include <eigen3/Eigen/Dense>

#include <cmath>
#include <iostream>
#include <vector>

using namespace Eigen;

// Compute the best approximation of the function $f$
// with Chebyshev polynomials.
// $\alpha$ is the output vector of coefficients.
template <typename Function>
VectorXd bestApproxCheb(const Function &f, const int n) {

    //TODO

}


int main() {
    int n;

	// Check the orthogonality of Chebyshev polynomials
    //TODO

	// Test the implementation for the given function
    //TODO

	return 0;
}

