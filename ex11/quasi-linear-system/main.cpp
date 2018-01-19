#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SparseCore>
#include <eigen3/Eigen/SparseLU>

#include <functional> // for std::function
#include <iostream>

using namespace Eigen;

// New version 1.1: using sparse matrices, dropping general nonlinear solver,
// using error instead of residual

//! \brief Implements a single step of the fixed point iteration
//! $x^{(k+1)} = A(x^{(k)})^{-1} * b$
//! \tparam func type of the lambda function implementing A(x)
//! \tparam Vector type for the vector b, x, x_new $\in \mathbf{R}^2$
//! \param[in] A lambda function implementing A(x)
//! \param[in] b rhs vector $b \in \mathbf{R}^n$
//! \param[in] x previous step $x^{(k)}$
//! \param[out] x_new next step $x^{(k+1)}$
template <class func, class Vector>
void fixed_point_step(const func & A, const Vector & b, const Vector & x,
                      Vector & x_new) {
    auto T = A(x);
    SparseLU<SparseMatrix<double>> AxLU;
    AxLU.analyzePattern(T);
    AxLU.factorize(T);
    x_new = AxLU.solve(b);
}

//! \brief Implements a single step of the Netwon iteration for $x^{(k+1)}$
//! Exploits Sherman-Morrison-Woodbury formula for fast inversion of rank-one
//! modification of a matrix.
//! \tparam func type of the lambda function implementing A(x)
//! \tparam Vector type for the vector b, x, x_new $\in \mathbf{R}^2$
//! \param[in] A lambda function implementing A(x)
//! \param[in] b rhs vector $b \in \mathbf{R}^n$
//! \param[in] x previous step $x^{(k)}$
//! \param[out] x_new next step in Newton iteration $x^{(k+1)}$
template <class func, class Vector>
void newton_step(const func & A, const Vector & b, const Vector & x,
                 Vector & x_new) {
    auto T = A(x);
    SparseLU<SparseMatrix<double>> AxLU;
    AxLU.analyzePattern(T);
    AxLU.factorize(T);

    auto Axinv_b = AxLU.solve(b);
    auto Axinv_x = AxLU.solve(x);

    double tmp = 1. / (x.norm() + x.dot(Axinv_x));

    x_new = Axinv_b + Axinv_x * x.transpose() * (x - Axinv_b) * tmp;
}

template <class func, class Vector>
void fixed_point_method(const func & A, const Vector & b, const double atol,
                        const double rtol, const int max_itr) {
    Vector x = b;
    Vector x_new = x;

    for (int i = 0; i < max_itr; ++i) {
        fixed_point_step(A, b, x, x_new);
        double r = (x - x_new).norm();

        std::cout << "[Step " << i << "] Error: " << r << std::endl;

        if (r < atol) {
            std::cout << "[CONVERGED] in " << i
                      << " iterations due to absolute tolerance" << std::endl
                      << std::endl;
            return;
        }

        if (r < rtol * x_new.norm()) {
            std::cout << "[CONVERGED] in " << i
                      << " iterations due to relative tolerance" << std::endl
                      << std::endl;
            return;
        }

        x = x_new;
    }

    std::cerr << "[DID NOT CONVERGE] reached max. iterations " << max_itr
              << std::endl << std::endl;
}

template <class func, class Vector>
void newton_method(const func & A, const Vector & b, const double atol,
                   const double rtol, const int max_itr) {
    Vector x = b;
    Vector x_new = x;

    for (int i = 0; i < max_itr; ++i) {
        newton_step(A, b, x, x_new);
        double r = (x - x_new).norm();

        std::cout << "[Step " << i << "] Error: " << r << std::endl;

        if (r < atol) {
            std::cout << "[CONVERGED] in " << i
                      << " iterations due to absolute tolerance" << std::endl
                      << std::endl;
            return;
        }

        if (r < rtol * x_new.norm()) {
            std::cout << "[CONVERGED] in " << i
                      << " iterations due to relative tolerance" << std::endl
                      << std::endl;
            return;
        }

        x = x_new;
    }

    std::cerr << "[DID NOT CONVERGE] reached max. iterations " << max_itr
              << std::endl << std::endl;
}


int main(void) {
    const double atol = 1e-13;
    const double rtol = 1e-11;
    const int max_itr = 100;

    // Define a test vector and test rhs and x0 = b
    const int n = 8;
    SparseMatrix<double> T(n,n);
    T.reserve(3);

    for (int i = 0; i < n; ++i) {
        if (i > 0) {
            T.insert(i,i-1) = 1;
        }

        T.insert(i,i) = 0;

        if (i < n-1) {
            T.insert(i,i+1) = 1;
        }
    }

    VectorXd b = VectorXd::Random(n);

    // Define a lambda function implementing A(x)
    auto A = [&T, n] (const VectorXd & x) -> SparseMatrix<double> & {
        double nrm = x.norm();

        for (int i = 0; i < n; ++i) {
            T.coeffRef(i,i) = 3 + nrm;
        }

        return T;
    };

    std::cout << "Fixed point method" << std::endl;
    fixed_point_method(A, b, atol, rtol, max_itr);

    std::cout << "Newton's method" << std::endl;
    newton_method(A, b, atol, rtol, max_itr);

    return 0;
}

