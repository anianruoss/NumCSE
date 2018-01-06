#include <eigen3/Eigen/Dense>
#include <iostream>


using namespace Eigen;

/* @brief
 * @param[in] z An $n$-dimensional vector containing one side of input data
 * @param[in] c An $n$-dimensional vector containing the other side of input data
 * @param[out] x The vector of parameters $(\alpha,\beta)$, intercept and slope of the line fitted
 */
VectorXd lsqEst(const VectorXd &z, const VectorXd &c) {
    assert (z.size() == c.size() && "z and c must have same size");
    const int n = z.size();

    MatrixXd A = MatrixXd::Zero(n,2);
    A.col(0) = z;
    A.col(1).head(n-1) += z.tail(n-1);
    A.col(1).tail(n-1) += z.head(n-1);

    VectorXd x(2);
    x = (A.transpose()*A).fullPivLu().solve(A.transpose()*c);

    return x;
}


int main() {
    const unsigned int n = 10;
    VectorXd z(n), c(n);

    for (size_t i=0; i<n; ++i) {
        z(i) = i+1;
        c(i) = n-i;
    }

    VectorXd x = lsqEst(z, c);

    std::cout << "alpha = " << x(0) << std::endl;
    std::cout << "beta = "  << x(1) << std::endl;

    return 0;
}

