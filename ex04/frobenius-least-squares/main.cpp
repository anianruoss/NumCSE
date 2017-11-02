#include <iostream>
#include <eigen3/Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>


using namespace Eigen;

MatrixXd min_frob(const VectorXd &z, const VectorXd &g) {
    const int n = z.size();

    MatrixXd C = kroneckerProduct(MatrixXd::Identity(n,n), z.transpose());

    MatrixXd CCT = -C*C.transpose();
    VectorXd p = CCT.fullPivLu().solve(g);

    VectorXd lamdba = -C.transpose()*p;

    MatrixXd M(n,n);

    for (int i = 0; i < n; ++i) {
        M.row(i) = lamdba.segment(n*i,n);
    }

    return M;
}


int main() {
    int n = 10;

    VectorXd z = VectorXd::Random(n);
    VectorXd g = VectorXd::Random(n);

    MatrixXd M_augmNormal = min_frob(z, g);;
    MatrixXd M_lagrange = g*z.transpose() / z.squaredNorm();

    std::cout << "Error = " << (M_augmNormal - M_lagrange).norm() << std::endl;

    return 0;
}

