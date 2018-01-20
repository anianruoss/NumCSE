#include <iostream>
#include <cassert>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

//! \brief Computes the slopes for natural cubic spline interpolation
//! \param[in] t vector of nodes
//! \param[in] y vector of values at the nodes
//! \param[out] c vector of slopes at the nodes
void natsplineslopes (const Vector &t, const Vector &y, Vector &c) {
	// Size check
	assert( ( t.size() == y.size() ) && "Error: mismatched size of t and y!");

	// n+1 is the number of conditions (t goes from t_0 to t_n)
	int n = t.size() - 1;

	Vector h(n);
	// Vector containing increments (from the right)
	for(int i = 0; i < n; ++i) {
		h(i) = t(i+1) - t(i);
		// Check that t is sorted
		assert( ( h(i) > 0 ) && "Error: array t must be sorted!");
	}

	// System matrix and rhs as in (3.5.9), we remove first and last row
	// (known by natural contition)
	Eigen::SparseMatrix<double> A(n+1,n+1);
	Vector b(n+1);

	// WARNING: sparse reserve space
	A.reserve(3);

	// Fill in natural conditions (3.5.10) for matrix
	A.coeffRef(0,0) = 2./h(0);
	A.coeffRef(0,1) = 1./h(0);
	A.coeffRef(n,n-1) = 1./h(n-1);
	A.coeffRef(n,n) = 2./h(n-1);

	// Reuse computation for rhs
	double bold = (y(1) - y(0)) / (h(0)*h(0));
	b(0) = 3.*bold; // Fill in natural conditions (3.5.10)
	// Fill matrix A and rhs b
	for(int i = 1; i < n; ++i) {
		// Fill in a
		A.coeffRef(i,i-1) = 1./h(i-1);
		A.coeffRef(i,i) = 2./h(i-1) + 2./h(i);
		A.coeffRef(i,i+1) = 1./h(i);

		// Reuse computation for rhs b
		double bnew = (y(i+1) - y(i)) / (h(i)*h(i));
		b(i) = 3. * (bnew + bold);
		bold = bnew;
	}
	b(n) = 3.*bold; // Fill in natural conditions (3.5.10)
	// Compress the matrix
	A.makeCompressed();

	// Factorize A and solve system A*c(1:end) = b
	Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;
	lu.compute(A);
	c = lu.solve(b);
}

//! \brief Computes the slopes for not-a-knot cubic spline interpolation
//! \param[in] t vector of nodes
//! \param[in] y vector of values at the nodes
//! \param[out] c vector of slopes at the nodes
void notknotsplines (const Vector &t, const Vector &y, Vector &c) {
	// Size check
	assert( ( t.size() == y.size() ) && "Error: mismatched size of t and y!");

	// n+1 is the number of conditions (t goes from t_0 to t_n)
	int n = t.size() - 1;

	Vector h(n);
	// Vector containing increments (from the right)
	for(int i = 0; i < n; ++i) {
		h(i) = t(i+1) - t(i);
		// Check that t is sorted
		assert( ( h(i) > 0 ) && "Error: array t must be sorted!");
	}

	// System matrix and rhs as in (3.5.9), we remove first and last row
	// (known by natural contition)
	Eigen::SparseMatrix<double> A(n+1,n+1);
	A.reserve(3*(n+1));
	Vector b(n+1);

	// Fill in not-a-knot conditions for matrix
	A.coeffRef(0,0) = 1. / (h(0)*h(0));
	A.coeffRef(0,1) = 1. / (h(0)*h(0)) - 1. / (h(1)*h(1));
	A.coeffRef(0,2) = - 1. / (h(1)*h(1));
	A.coeffRef(n,n-2) = 1. / (h(n-2)*h(n-2));
	A.coeffRef(n,n-1) = 1. / (h(n-2)*h(n-2)) - 1. / (h(n-1)*h(n-1));
	A.coeffRef(n,n) = - 1. / (h(n-1)*h(n-1));

	// Fill matrix A and rhs b
	// Fill in not-a-knot conditions
	b(0) = 2.*((y(1)-y(0))/std::pow(h(0),3) + (y(1)-y(2))/std::pow(h(1),3));
	b(n) = 2.*(y(n-1)-y(n-2))/std::pow(h(n-2),3);
	b(n) += 2.*(y(n-1)-y(n))/std::pow(h(n-1),3);

	// Reuse computation for rhs
	double bold = (y(1) - y(0)) / (h(0)*h(0));

	for(int i = 1; i < n; ++i) {
		// Fill in a
		A.coeffRef(i,i-1) = 1./h(i-1);
		A.coeffRef(i,i) = 2./h(i-1) + 2./h(i);
		A.coeffRef(i,i+1) = 1./h(i);

		// Reuse computation for rhs b
		double bnew = (y(i+1) - y(i)) / (h(i)*h(i));
		b(i) = 3. * (bnew + bold);
		bold = bnew;
	}

	// Compress the matrix
	A.makeCompressed();

	// Factorize A and solve system A*c(1:end) = b
	Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;
	lu.compute(A);
	c = lu.solve(b);
}


int main() {
	const size_t n = 10;
	Vector y = Vector::Random(n);
	Vector t = Vector::Random(n);
	Vector c_nat = Vector::Zero(n);
	Vector c_knot = Vector::Zero(n);

	std::sort(t.data(), t.data()+t.size(), [] (double a, double b) {
			return a < b;
			});

	natsplineslopes(t, y, c_nat);
	notknotsplines(t, y, c_knot);

	std::cout << "Natural spline slopes" << std::endl;
	std::cout << c_nat.tranpose() << std::endl << std::endl;
	std::cout << "Not-a-knot spline slopes" << std::endl;

	return 0;
}

