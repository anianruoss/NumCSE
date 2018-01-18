#include <eigen3/Eigen/Dense>

#include <cmath>
#include <iostream>
#include <limits>


using namespace Eigen;

/* @brief Newton's method to approximate $x^{(0)}$
 * @param[in] x0_ Initial guess
 * @param[out] x0 Final estimation of $x^{(0)}$, given convergence of Newton's method
 */
double newton_arctan(double x0_) {
    //TODO
}


int main() {
	// Initialization
    double x0_ = 2; // Initial guess

  // Netwon's method
    double x0 = newton_arctan(x0_);

	return 0;
}

