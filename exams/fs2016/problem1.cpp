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

    // TODO: problem 1c: implement a Gauss-Seidel solver for Ax = b

    return;
}


int main(int, char**) {
    // Build matrix and r.h.s.
    unsigned int n = 9; // A is n x n, b and x have length n

    Matrix A = Matrix::Zero(n,n);
    for(unsigned int i = 0; i < n; ++i) {
        if(i > 0) A(i,i-1) = 2;
        A(i,i) = 3;
        if(i < n-1) A(i,i+1) = 1;
    }
    Vector b = Vector::Constant(n,1);

    //// PROBLEM 1d
    std::cout << "*** PROBLEM 1d:" << std::endl;

    // TODO: problem 1d: test the code GSIt using the given data

    double residual = 0.; // TODO: problem 1d: compute residual

    std::cout << "Residual = " << residual << std::endl;

	return 0;
}

