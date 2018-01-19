#include <iostream>

#include <Eigen/Dense>
#include <Eigen/SVD>

using namespace Eigen;

/*! @brief Compute the polar decomposition of $\mathbf{M} = \mathbf{A}\mathbf{Q}$.
 *! @param[in] M  A square matrix.
 *! @return       The pair of matrices $(\mathbf{A},\mathbf{Q})$.
 */
std::pair<MatrixXd,MatrixXd> polar(const MatrixXd & M) {
    assert(M.rows() == M.cols() && "M must be square!");
    unsigned int n = M.rows();

    JacobiSVD<MatrixXd> svd(M, ComputeThinU | ComputeThinV);
    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV();
    MatrixXd Sigma = svd.singularValues().asDiagonal();

    return std::make_pair(U * Sigma * U.transpose(), U * V.transpose());
}

/*! @brief Test routine for the function 'polar'.
 *! @param[in]  M   Square matrix.
 *! @return         'true' of 'polar' returns a polar decomposition of $\mathb{M}$.
 */
bool testPolar(const MatrixXd& M) {
    auto AQ = polar(M);
    MatrixXd A = AQ.first;
    MatrixXd Q = AQ.second;

    assert(A.rows() == M.rows() && A.cols() == M.cols() &&
           "A must have same dimension of square matrix M");
    assert(Q.rows() == M.rows() && Q.cols() == M.cols() &&
           "Q must have same dimension of square matrix M");

    // Is $\VA$ symmetric?
    if (!A.isApprox(A.transpose())) {
        return false;
    }
    // Is positive semi-definite?
    LLT<MatrixXd> llt(A);
    if (llt.info() == NumericalIssue) {
        return false;
    }
    // Is $\VQ$ orthogonal?
    MatrixXd I = MatrixXd::Identity(M.rows(),M.cols());
    MatrixXd QtQ = Q.transpose() * Q;
    MatrixXd QQt = Q * Q.transpose();
    if (!QtQ.isApprox(I) || !QQt.isApprox(I)) {
        return false;
    }
    // Is $\VA\VQ = \VM$?
    if (!M.isApprox(A * Q)) {
        return false;
    }
    // All tests completed, 'polar' represents a polar decomposition.
    return true;
}


int main() {
    // Test matrix
    unsigned n = 3;
    MatrixXd M(n,n);
    M << 1,  2,  3,
    2,  1,  3,
    6,  3, 11;

    auto AQ = polar(M);
    MatrixXd A = AQ.first;
    MatrixXd Q = AQ.second;
    std::cout << "Matrix A is:" << std::endl << A << std::endl << std::endl;
    std::cout << "Matrix Q is:" << std::endl << Q << std::endl << std::endl;
    std::cout << "Is the polar decomposition implemented correctly? "
              << std::boolalpha << testPolar(M) << std::endl;

    return 0;
}

