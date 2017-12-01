#include <eigen3/Eigen/Dense>
#include <mgl2/mgl.h>
#include <iostream>


using namespace Eigen;

// returns the values of the piecewise linear interpolant in evalT.
VectorXd evalPiecewiseInterp(const VectorXd &T, const VectorXd &Y, const VectorXd &evalT) {
    const int n = evalT.size();
    VectorXd out(n);

    // TODO: Implement me

    return out;
}

double maxInterpError(double a, VectorXd T, VectorXd evalT) {
    double maxError = 0;

    // TODO: Implement me

    return maxError;
}


int main() {

    // TODO: Implement me

    return 0;
}

