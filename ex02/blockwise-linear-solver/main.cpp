#include <iostream>
#include <Eigen/Dense>


using namespace Eigen;


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

// Point (f)
void solve_blockGauss(const MatrixXd &R, const VectorXd &u, const VectorXd &v,
                      const VectorXd &b, VectorXd &x) {
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

    x1 = A.fullPivLu().solve(b);
    std::cout << (A*x1 - b).norm() << std::endl;

    solve_blockLU(R,u,v,b,x2);
    std::cout << (A*x2 - b).norm() << std::endl;

    solve_blockGauss(R, u, v, b, x3);

    return 0;
}

