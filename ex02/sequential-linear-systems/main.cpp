#include "timer.h"
#include <iostream>
#include <eigen3/Eigen/Dense>


using namespace Eigen;

// Complexity: O(mn^3)
void solve_naive(const MatrixXd &A, const MatrixXd &b, MatrixXd &X) {
    for (int i = 0; i < b.cols(); ++i) {
        X.col(i) = A.fullPivLu().solve(b.col(i));
    }
}

// Complexity: O(n^3 + mn^2)
void solve_LU(const MatrixXd &A, const MatrixXd &b, MatrixXd &X) {
    FullPivLU<MatrixXd> lu(A);

    for (int i = 0; i < b.cols(); ++i) {
        X.col(i) = lu.solve(b.col(i));
    }
}

void solve_eigenLU(const MatrixXd &A, const MatrixXd &b, MatrixXd &X) {
    X = A.fullPivLu().solve(b);
}


int main() {
    int n = 3;
    int m = 5;

    srand(time(NULL));

    MatrixXd A = MatrixXd::Random(n,n);
    MatrixXd b = MatrixXd::Random(n,m);
    MatrixXd X1(n,m);
    MatrixXd X2(n,m);
    MatrixXd X3(n,m);

    Timer t;

    std::cout << "--> Naive:" << std::endl;
    t.start();
    solve_naive(A,b,X1);
    t.stop();
    std::cout << "Error: " << (A*X1 - b).norm() << std::endl;
    std::cout << "Time:  " << t.duration() << std::endl;

    std::cout << std::endl << "--> LU-decomposition:" << std::endl;
    t.reset();
    t.start();
    solve_LU(A,b,X2);
    t.stop();
    std::cout << "Error: " << (A*X2 - b).norm() << std::endl;
    std::cout << "Time:  " << t.duration() << std::endl;

    std::cout << std::endl << "--> Eigen-LU: " << std::endl;
    t.reset();
    t.start();
    solve_eigenLU(A,b,X3);
    t.stop();
    std::cout << "Error: " << (A*X3 - b).norm() << std::endl;
    std::cout << "Time:  " << t.duration() << std::endl;

    return 0;
}

