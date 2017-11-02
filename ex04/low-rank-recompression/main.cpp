#include <iostream>
#include <limits>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/SVD>


using namespace Eigen;

void factorize(const MatrixXd & X, int k, MatrixXd & A, MatrixXd & B) {
    JacobiSVD<MatrixXd> svd(X, ComputeThinU | ComputeThinV);
    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV();
    VectorXd sv = svd.singularValues();

    A = U.array().rowwise() * sv.transpose().array();
    B = V;
}

void svd_AB(const MatrixXd & A, const MatrixXd & B,
            MatrixXd & U, MatrixXd & S, MatrixXd & V) {
	assert(A.cols() == B.cols() && "A and B need same amount of columns");
	using index_t = MatrixXd::Index;
	const index_t m = A.rows(), k = A.cols(), n = B.rows();

	HouseholderQR<MatrixXd> qr_A(A);
	MatrixXd Qa = qr_A.householderQ() * MatrixXd::Identity(m, std::min(m,k));
	MatrixXd Ra = MatrixXd::Identity(std::min(m,k), m) * qr_A.matrixQR().triangularView<Upper>();

	HouseholderQR<MatrixXd> qr_B(B);
	MatrixXd Qb = qr_B.householderQ() * MatrixXd::Identity(n, std::min(n,k));
	MatrixXd Rb = MatrixXd::Identity(std::min(n,k), n) * qr_B.matrixQR().triangularView<Upper>();

	JacobiSVD<MatrixXd> svd(Ra*Rb.transpose(), ComputeThinU | ComputeThinV);
	U = Qa * svd.matrixU();
	V = Qb * svd.matrixV();
	VectorXd sv = svd.singularValues();
	S = sv.asDiagonal();
}

void rank_k_approx(const MatrixXd &Ax, const MatrixXd &Ay,
                   const MatrixXd &Bx, const MatrixXd &By,
                   MatrixXd &Az, MatrixXd &Bz) {
}


int main() {
    int m = 3;
    int n = 2;
    int k = 2;

	std::cout << "--> Factorizing" << std::endl;
    MatrixXd X(m,n);
    X << 5, 0, 2, 1, 7, 4;

    MatrixXd A, B;
    factorize(X, k, A, B);

    std::cout << "A =" << std::endl << A << std::endl << std::endl;
    std::cout << "B =" << std::endl << B << std::endl << std::endl;
	std::cout << "Error: " << (X - A*B.transpose()).norm() << std::endl;
	std::cout << std::endl;

	std::cout << "--> Low Rank SVD" << std::endl;
    A.resize(m,k);
    B.resize(n,k);
    A << 2, 1, 2, 3, 6, 1;
    B << 4, 4, 5, 0;
    MatrixXd U, S, V;

    svd_AB(A, B, U, S, V);
    std::cout << "U =" << std::endl << U << std::endl << std::endl;
    std::cout << "S =" << std::endl << S << std::endl << std::endl;
    std::cout << "V =" << std::endl << V << std::endl << std::endl;
	std::cout << "Error: " << (A*B.transpose() - U*S*V.transpose()).norm() << std::endl;
	std::cout << std::endl;

    MatrixXd Ax(m,k), Ay(m,k), Bx(n,k), By(n,k), Az, Bz;
    Ax << 1,  0, 9, 2, 6, 3;
    Ay << 8, -2, 3, 4, 5, 8;
    Bx << 2, 1, 2, 3;
    By << 4, 4, 5, 0;

    rank_k_approx(Ax, Ay, Bx, By, Az, Bz);

    std::cout << "Az =" << std::endl << Az << std::endl;
    std::cout << "Bz =" << std::endl << Bz << std::endl;

    return 0;
}

