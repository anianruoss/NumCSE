#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>

struct QuadRule {
    Eigen::VectorXd nodes, weights;
};

void gaussquad(const unsigned n, QuadRule& qr) {
    // Initialize bidiagonal matrix \Blue{$\VJ_n$}
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n, n);
    for (unsigned i = 1; i < n; ++i) {
        const double b = i/std::sqrt(4.*i*i - 1.); // \Label[line]{gw:3}
        M(i,i-1) = b;
        M(i-1,i) = b;                // \Label[line]{gw:3x}
    }
    // \eigen's built-in helper class for eigenvalue problems
    // (use method for symmetric matrices, exploiting the structure)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(M);

    qr.nodes = eig.eigenvalues();
    qr.weights = 2*eig.eigenvectors().topRows<1>().array().pow(2); // \Label[line]{gw:4}
}

