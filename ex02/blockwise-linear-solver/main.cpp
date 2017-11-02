#include "timer.h"
#include <iostream>
#include <eigen3/Eigen/Dense>


using namespace Eigen;

// Complexity: O(n^2)
VectorXd solve_backSub(const MatrixXd &R, const VectorXd &c) {
    int n = R.rows();
    VectorXd y(n);

    for (int i = n-1; i >= 0; --i) {
        y(i) = c(i);

        for (int j = n-1; j > i; --j) {
            y(i) -= R(i,j) * y(j);
        }

        y(i) /= R(i,i);
    }

    return y;
}

// Complexity: O(n^2)
void solve_blockLU(const MatrixXd &R, const VectorXd &u, const VectorXd &v,
                   const VectorXd &b, VectorXd &x) {
    unsigned int n = R.rows();
    VectorXd y(n+1);

    // Solve Ly=b by forward substitution
    y.head(n) = b.head(n);
    y(n) = b(n) - u.transpose()*solve_backSub(R, y.head(n));

    // Solve Ux=y by backward substitution
    MatrixXd U(n+1,n+1);
    U << R, v, VectorXd::Zero(n).transpose(), -u.transpose()*solve_backSub(R,v);

    x = solve_backSub(U,y);
}

// Complexity: O(n^2)
void solve_blockGauss(const MatrixXd &R, const VectorXd &u, const VectorXd &v,
                      const VectorXd &b, VectorXd &x) {
    unsigned int n = R.rows();

    double s = -u.transpose()*solve_backSub(R,v);
    double bs = b(n) - u.transpose()*solve_backSub(R,b.head(n));

    x.head(n) = solve_backSub(R, (b.head(n) - v*bs/s));
    x(n) = bs/s;
}


int main() {
    unsigned int n = 3;

    MatrixXd R = MatrixXd::Random(n,n);
    R = R.triangularView<Upper>();
    VectorXd u = VectorXd::Random(n);
    VectorXd v = VectorXd::Random(n);
    VectorXd b = VectorXd::Random(n+1);
    VectorXd x1(n+1);
    VectorXd x2(n+1);
    VectorXd x3(n+1);

    MatrixXd A(n+1,n+1);
    A << R , v, u.transpose(), 0;
    Timer t;

    std::cout << "--> EigenLU" << std::endl;
    t.start();
    x1 = A.fullPivLu().solve(b);
    t.stop();
    std::cout << "Error: " << (A*x1 - b).norm() << std::endl;
    std::cout << "Time:  " << t.duration() << std::endl << std::endl;
    t.reset();

    std::cout << "--> BlockwiseLU" << std::endl;
    t.start();
    solve_blockLU(R,u,v,b,x2);
    t.stop();
    std::cout << "Error: " << (A*x2 - b).norm() << std::endl;
    std::cout << "Time:  " << t.duration() << std::endl << std::endl;
    t.reset();

    std::cout << "--> Gaussian Elimination" << std::endl;
    t.start();
    solve_blockGauss(R, u, v, b, x3);
    t.stop();
    std::cout << "Error: " << (A*x3 - b).norm() << std::endl;
    std::cout << "Time:  " << t.duration() << std::endl;

    return 0;
}

