#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

#include <iostream>

using namespace Eigen;

//!
//! \brief buildDistanceLSQMatrix Efficiently build the system matrix
//! mapping positions to distances.
//! \param n Number of points (including $x_1$).
//! \return The system matrix $A$.
//!

SparseMatrix<double> buildDistanceLSQMatrix(int n) {
    SparseMatrix<double> A(n*(n-1)/2, n-1);

    // TODO efficiently build and return the system matrix A
    A.makeCompressed();
    return A;
}


//!
//! \brief estimatePointsPositions Return positions (without $x_1$).
//! The coordinate $x_1$ is assumed $x_1 = 0$.
//! \param D An $n \times n$ anti-symmetric matrix of distances.
//! \return Vector of positions $x_2, \dots, x_n$.
//!

VectorXd estimatePointsPositions(const MatrixXd& D) {

    VectorXd x;

    // TODO compute and return the vector of positions $[x_2, \dots, x_n]$ given
    // the matrix of distances and assuming $x_1 = 0$.
    // Make the implementation as efficient as possible.

    return x;
}


int main() {

    int n = 5;

    // PART 1: build and print system matrix A
    std::cout << "**** PART 1 ****" << std::endl;
    std::cout << "The Matrix A is:"
              << std::endl
              << buildDistanceLSQMatrix(n)
              << std::endl;

    // PART 2: solve the LSQ system and find positions
    std::cout << "**** PART 2 ****" << std::endl;
    // Vector of positions
    n = 5;

    // Build D
    MatrixXd D(n,n);
    D << 0,   -2.1, -3,   -4.2, -5,
    2.1,  0,   -0.9, -2.2, -3.3,
    3,    0.9,  0,   -1.3, -1.1,
    4.2,  2.2,  1.3,  0,   -1.1,
    5,    3.3,  1.1,  1.1,  0;
    std::cout << "The matrix D is:"
              << std::endl
              << D
              << std::endl;

    // Find out positions
    VectorXd x_recovered = estimatePointsPositions(D);
    std::cout << "Positions [x_2, ..., x_n] obtained from the LSQ system:"
              << std::endl << x_recovered << std::endl;

    return 0;
}

