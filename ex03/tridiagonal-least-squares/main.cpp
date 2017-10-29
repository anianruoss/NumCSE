#include <iostream>
#include <Eigen/Dense>


using namespace Eigen;

/* @brief
 * @param[in] z An $n$-dimensional vector containing one side of input data
 * @param[in] c An $n$-dimensional vector containing the other side of input data
 * @param[out] x The vector of parameters $(\alpha,\beta)$, intercept and slope of the line fitted
 */
VectorXd lsqEst(const VectorXd &z, const VectorXd &c) {
    assert (z.size() == c.size() && "z and c must have same size");
    int n = z.size();

	VectorXd z1(n);
	z1(0) = z(1);
	z1(n-1) = z(n-2);
	
	for (int i = 1; i < n-1; ++i) {
		z1(i) = z(i-1) + z(i+1);
	}

	MatrixXd A(n,2);
	A << z, z1;

    VectorXd x(2);
	x = (A.transpose()*A).fullPivLu().solve(A.transpose()*c);
	
    return x;
}

int main() {
    unsigned int n = 10;
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

