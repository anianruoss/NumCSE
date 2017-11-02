#include <iostream>
#include <limits>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/QR>


using namespace Eigen;

/* @brief QR decomposition from Cholesky decomposition
 * @param[in] A An $m \times n$ matrix
 * @param[out] R The upper triangular matrix from the QR decomposition of $A$
 * @param[out] Q The orthogonal matrix from the QR decomposition of $A$
 */
void CholeskyQR(const MatrixXd & A, MatrixXd & R, MatrixXd & Q) {
    MatrixXd AtA = A.transpose()*A;
    LLT<MatrixXd> chol = AtA.llt();
    MatrixXd L = chol.matrixL();
    R = L.transpose();
    Q = L.triangularView<Lower>().solve(A.transpose()).transpose();
}

/* @brief Direct QR decomposition
 * @param[in] A An $m \times n$ matrix
 * @param[out] R The upper triangular matrix from the QR decomposition of $A$
 * @param[out] Q The orthogonal matrix from the QR decomposition of $A$
 */
void DirectQR(const MatrixXd & A, MatrixXd & R, MatrixXd & Q) {
    using index_t = MatrixXd::Index;
    const index_t m = A.rows(), n = A.cols();
    HouseholderQR<MatrixXd> qr(A);
    Q = qr.householderQ() * MatrixXd::Identity(m,std::min(m,n));
    R = MatrixXd::Identity(std::min(m,n), m) * qr.matrixQR().triangularView<Upper>();
}

int main() {
    bool precision;
    std::cout << "0 for normal QR-Decomposition \n1 for precision test" << std::endl;
	std::cout << std::endl;
    std::cin >> precision;
	std::cout << std::endl;

    size_t m = 3;
    size_t n = 2;

    MatrixXd A(m,n);
    if (precision) {
        double epsilon = std::numeric_limits<double>::denorm_min();
        A << 1, 1, 0.5*epsilon, 0, 0, 0.5*epsilon;
    } else {
        A << 3, 5, 1, 9, 7, 1;
    }
    std::cout << "A =" << std::endl << A << "\n" << std::endl;

    MatrixXd R, Q;

    std::cout << "--> Cholesky QR-Decomposition" << std::endl;
    CholeskyQR(A, R, Q);
    std::cout << "R =" << std::endl << R << std::endl << std::endl;
    std::cout << "Q =" << std::endl << Q << std::endl << std::endl;
    std::cout << "Error" << std::endl
              << "||A - Q*R|| = " << (A - Q*R).norm() << std::endl;
    std::cout << std::endl << std::endl;


    std::cout << "--> Householder QR-Decomposition" << std::endl;
    DirectQR(A, R, Q);
    std::cout << "R =" << std::endl << R << std::endl << std::endl;
    std::cout << "Q =" << std::endl << Q << std::endl << std::endl;
    std::cout << "Error" << std::endl
              << "||A - Q*R|| = " << (A - Q*R).norm() << std::endl;

    return 0;
}

