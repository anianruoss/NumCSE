#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;


void solve_naive(const MatrixXd &A, const MatrixXd &b, MatrixXd &X) {
    X = A.fullPivLu().solve(b);
}


void solve_LU(const MatrixXd &A, const MatrixXd &b, MatrixXd &X) {
    Eigen::FullPivLU<MatrixXd> lu(A);

    // solve Lz = Pb
    MatrixXd Pb = lu.permutationP() * b;
    MatrixXd L = MatrixXd::Identity(A.cols(),A.cols());
    L.triangularView<StrictlyLower>() = lu.matrixLU();
    MatrixXd z = L.inverse() * Pb;

    // solve UQ^-1x = z
    MatrixXd U = lu.matrixLU().triangularView<Upper>();
    MatrixXd R = U * lu.permutationQ().inverse();
    X = R.inverse() * z;
}


int main() {
    int n = 3;
    int m = 5;

    srand(time(NULL));

    MatrixXd A = MatrixXd::Random(n,n);
    MatrixXd b = MatrixXd::Random(n,m);
    MatrixXd X1(n,m);
    MatrixXd X2(n,m);

    std::cout << "--> Naive:" << std::endl;
    solve_naive(A,b,X1);
    std::cout << "A*x = " << std::endl << A*X1 << std::endl;
    std::cout << std::endl << "b = " << std::endl << b << std::endl;
    std::cout << std::endl << "Error: " <<
              std::endl << (A*X1 - b).norm() << std::endl;

    std::cout << std::endl << "--> LU-decomposition:" << std::endl;
    solve_LU(A,b,X2);
    std::cout << "A*x = " << std::endl << A*X2 << std::endl;
    std::cout << std::endl << "b = "	<< std::endl << b << std::endl;
    std::cout << std::endl << "Error: " <<
              std::endl << (A*X2 - b).norm() << std::endl;

    return 0;
}

