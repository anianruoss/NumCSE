#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseLU>

#include <iostream>
#include <vector>

using Triplet = Eigen::Triplet<double>;
using Triplets = std::vector<Triplet>;

using Vector = Eigen::VectorXd;
using Matrix = Eigen::SparseMatrix<double>;

//! \brief Efficiently construct the sparse matrix A given c, i_0 and j_0
//! \param[in] c contains entries c_i for matrix A
//! \param[in] i0 row index i_0
//! \param[in] j0 column index j_0
//! \return Sparse matrix A
Matrix buildA(const Vector & c, const unsigned int i0, const unsigned int j0) {
    assert(i0 > j0);

    const unsigned int n = c.size() + 1;
    Matrix A(n,n);
    Triplets triplets;
    triplets.reserve(2*n);

    for (unsigned int i = 0; i < n; ++i) {
        triplets.push_back(Triplet(i, i, 1.));
    }

    for (unsigned int i = 0; i < n-1; ++i) {
        triplets.push_back(Triplet(i, i+1, c(i)));
    }

    triplets.push_back(Triplet(i0-1, j0-1, 1.));

    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    return A;
}

Vector backSub(const Vector & c, const Vector & b) {
    const size_t n = c.size() + 1;
    Vector x = Vector::Zero(n);
    x(n-1) = b(n-1);

    for (size_t i = 1; i < n; ++i) {
        x(n-1-i) = b(n-1-i) - x(n-i) * c(n-1-i);
    }

    return x;
}

//! \brief Solve the system Ax = b with optimal complexity O(n)
//! \param[in] c contains entries c_i for matrix A
//! \param[in] b r.h.s. vector
//! \param[in] i0 row index
//! \param[in] j0 column index
//! \return Solution x, s.t. Ax = b
Vector solveLSE(const Vector & c, const Vector & b, unsigned int i0,
                unsigned int j0) {
    assert(c.size() == b.size()-1 && "Size mismatch!");
    assert(i0 > j0);

    Vector u = Vector::Zero(b.size());
    Vector v = Vector::Zero(b.size());
    u(i0-1) = 1.;
    v(j0-1) = 1.;

    Vector Ainvb = backSub(c, b);
    Vector Ainvu = backSub(c, u);

    return Ainvb -  Ainvu * v.dot(Ainvb) / (1. + v.dot(Ainvu));

}


int main(int, char**) {
    // Setup data for problem
    unsigned int n = 15; // A is n x n matrix, b has length x

    unsigned int i0 = 6, j0 = 4;

    Vector b = Vector::Random(n); // Random vector for b
    Vector c = Vector::Random(n-1); // Random vector for c

    // Solve sparse system using sparse LU and our own routine
    Matrix A = buildA(c, i0, j0);
    A.makeCompressed();
    Eigen::SparseLU<Matrix> splu;
    splu.analyzePattern(A);
    splu.factorize(A);

    std::cout << "Error: " << (splu.solve(b) - solveLSE(c, b, i0, j0)).norm()
              << std::endl;

    return 0;
}

