#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace Eigen;


/* \brief Performs Gram-Schidt orthonormalization
 * Given a matrix $\mathbf{A}$ of linearly independent columns,
 * returns the result of a Gram-Schmidt orthonormalization.
 * Unstable GS algorithm: output is prone to cancellation issues.
 * \param[in] $\mathbf{A}$ Matrix of linearly independent columns
 * \return Matrix with ONB of $span(a_1, \cdots, a_n)$ as columns
 */
MatrixXd gram_schmidt(const MatrixXd & A) {
    // We create a matrix Q with the same size and data of A
    MatrixXd Q(A);

    Q.col(0).normalize();

    for (unsigned int j = 1; j < A.cols(); ++j) {
        Q.col(j) -= Q.leftCols(j) * (Q.leftCols(j).transpose() *A.col(j));

        // Normalize vector, if possible, (otherwise it means that
        // columns of $\mathbf{A}$ are almost linear dependant)
        double eps = std::numeric_limits<double>::denorm_min();
        if (Q.col(j).norm() <= eps * A.col(j).norm()) {
            std::cerr << "Gram-Schmidt failed because A has "
                      << "(almost) linear dependant columns." << std::endl;
            break;
        } else {
            Q.col(j).normalize();
        }
    }

    return Q;
}


int main(void) {
    // Orthonormality test
    unsigned int n = 9;
    MatrixXd A = MatrixXd::Random(n,n);

    MatrixXd Q = gram_schmidt(A);
    double err = (Q.transpose()*Q - MatrixXd::Identity(n,n)).norm();

    std::cout << "Error is: " << err << std::endl;

    // If error is too big, we exit with error
    double eps = std::numeric_limits<double>::denorm_min();
    exit(err < eps);

    return 0;
}

