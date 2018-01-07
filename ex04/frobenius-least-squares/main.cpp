#include <eigen3/Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <iostream>


using namespace Eigen;

MatrixXd minFrob(const VectorXd &z, const VectorXd &g) {
    const int n = z.size();

    MatrixXd C = kroneckerProduct(MatrixXd::Identity(n,n), z.transpose());
    MatrixXd CCT = -C*C.transpose();

    VectorXd p = CCT.fullPivLu().solve(g);
    VectorXd lamdba = -C.transpose()*p;

    MatrixXd M = Map<MatrixXd>(lamdba.data(), n, n).transpose();

    return M;
}


int main() {
    int n = 100;

    VectorXd z = VectorXd::Random(n);
    VectorXd g = VectorXd::Random(n);

    MatrixXd M_augmNormal = minFrob(z, g);
    MatrixXd M_lagrange = g*z.transpose() / z.squaredNorm();

    std::cout << "Error = " << (M_augmNormal - M_lagrange).norm() << std::endl;

    return 0;
}

