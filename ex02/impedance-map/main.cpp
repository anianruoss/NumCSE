#include <iostream>
#include <iomanip>
#include <eigen3/Eigen/Dense>


using namespace Eigen;

// modifies ARx so that Rx = newRx
void changeRx(MatrixXd &ARx, double newRx) {
    ARx(13,14) = -1/newRx;
    ARx(14,13) = -1/newRx;
    ARx(13,13) = 3 + 1/newRx;
    ARx(14,14) = 4 + 1/newRx;
}

double calc_impedance(const MatrixXd &A1inv, const VectorXd &b, double Rx,
                      double V) {
    double f = 1/Rx;
    VectorXd u = VectorXd::Zero(b.size());
    VectorXd v = VectorXd::Zero(b.size());

    u(13) = 1;
    u(14) = -1;
    v(13) = f-1;
    v(14) = 1-f;

    double alpha = 1 + (v.transpose() * A1inv * u);
    VectorXd y = A1inv * b;
    VectorXd x = y - (A1inv * u * v.transpose() * y)/alpha;

    return V / (V - x(5));
}


int main() {
    MatrixXd ARx(15,15), A1inv(15,15);
    VectorXd b = VectorXd::Zero(15);
    VectorXd x = VectorXd::Zero(15);

    // source term from node 16
    int V = 1;
    b(5) = V;

    // initialize ARx with Rx = 1
    ARx <<
        2,-1, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        -1, 4,-1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0,
        0,-1, 3,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,
        0, 0,-1, 3, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0,-1,
        -1,-1, 0, 0, 4, 0,-1, 0, 0, 0, 0, 0, 0,-1, 0,
        0, 0, 0,-1, 0, 4, 0, 0,-1, 0, 0, 0, 0, 0,-1,
        0, 0, 0, 0,-1, 0, 4, 0, 0,-1,-1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 4,-1, 0, 0,-1,-1, 0,-1,
        0, 0, 0, 0, 0,-1, 0,-1, 3, 0, 0, 0,-1, 0, 0,
        0, 0, 0, 0, 0, 0,-1, 0, 0, 2,-1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,-1, 0, 0,-1, 4,-1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,-1, 0, 0,-1, 3,-1, 0, 0,
        0, 0, 0, 0, 0, 0, 0,-1,-1, 0, 0,-1, 3, 0, 0,
        0,-1, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 4,-1,
        0, 0,-1,-1, 0,-1, 0,-1, 0, 0, 0, 0, 0,-1, 5;

    A1inv = ARx.inverse();

    std::cout << "--> Calculating impedance for:" << std::endl;

    for (unsigned int i = 0; i <= 10; ++i) {
        std::cout << std::setw(7) << "Rx = 2^" << std::setw(2) << i
                  << std::setw(10) << calc_impedance(A1inv, b, 1 << i, V)
				  << std::endl;
    }

    return 0;
}

