#include <iostream>
#include <Eigen/Dense>

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

//! \brief Use symmetric Gauss-Seidel iterations to approximate the solution of the system Ax = b
//! \param[in] A  = L + D + U, system matrix to be decomposed, must be invertible
//! \param[in] b r.h.s. vector
//! \param[in,out] x initial guess as input and last value of iteration as output
//! \param[in] rtol relative tolerance for termination criteria
void GSIt(const Matrix & A, const Vector & b, Vector & x, double rtol) {
    Eigen::TriangularView<Matrix, Eigen::Upper> UpD(A);
    Eigen::TriangularView<Matrix, Eigen::Lower> LpD(A);
    Eigen::TriangularView<Matrix, Eigen::StrictlyUpper> U(A);
    Eigen::TriangularView<Matrix, Eigen::StrictlyLower> L(A);

    Vector x_new = x;

    while (true) {
        x_new = UpD.solve(b) - UpD.solve(L*LpD.solve(b - U*x));

        double r = (x_new - x).norm();

        if (r < rtol * x_new.norm()) {
            return;
        }

        x = x_new;
    }
}


int main(int, char**) {
    const unsigned int n = 9; // A is n x n, b and x have length n

    Matrix A = Matrix::Zero(n,n);
    for (unsigned int i = 0; i < n; ++i) {
        if (i > 0) A(i,i-1) = 2;
        A(i,i) = 3;
        if (i < n-1) A(i,i+1) = 1;
    }
    Vector b = Vector::Constant(n,1);

    std::cout << "*** PROBLEM 1d:" << std::endl;
    Vector x = Vector::Random(n);
    GSIt(A, b, x, 10e-8);

    double residual = (A*x - b).norm();

    std::cout << "Residual = " << residual << std::endl;

    return 0;
}

