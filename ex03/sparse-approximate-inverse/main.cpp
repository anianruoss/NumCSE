#include "timer.h"
#include <iostream>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/KroneckerProduct>


using namespace Eigen;
using index_t = int;

SparseMatrix<double> spai(SparseMatrix<double> &A) {
    assert(A.rows() == A.cols() && "Matrix must be square!");
    unsigned int N = A.rows();

    A.makeCompressed();

    // obtain pointers to data of A
    double *valPtr = A.valuePtr();
    index_t *innPtr = A.innerIndexPtr();
    index_t *outPtr = A.outerIndexPtr();

    // create vector for triplets of B and reserve enough space
    std::vector<Triplet<double>> B_triplets;
    B_triplets.reserve(A.nonZeros());

    // project P(A) onto P(a_i) and compute b_i
    for (unsigned int i = 0; i < N; ++i) {
        index_t nnz_i = outPtr[i+1] - outPtr[i];
        if (nnz_i == 0) continue;

        SparseMatrix<double> C(N, nnz_i);
        std::vector<Triplet<double>> C_triplets;
        C_triplets.reserve(nnz_i*nnz_i);

        for (int k = outPtr[i]; k < outPtr[i+1]; ++k) {
            index_t row_k = innPtr[k];
            index_t nnz_k = outPtr[row_k+1] - outPtr[row_k];

            for (int l = 0; l < nnz_k; ++l) {
                int innIdx = outPtr[row_k] + l;
                C_triplets.emplace_back(Triplet<double> (innPtr[innIdx], k-outPtr[i], valPtr[innIdx]));
            }
        }

        C.setFromTriplets(C_triplets.begin(), C_triplets.end());
        C.makeCompressed();


        SparseMatrix<double> S = C.transpose()*C;
        VectorXd xt = C.row(i).transpose();
        SparseLU<SparseMatrix<double>> spLU(S);
        VectorXd b = spLU.solve(xt);

        for (unsigned int k = 0; k < b.size(); ++k) {
            B_triplets.emplace_back(Triplet<double> (innPtr[outPtr[i]+k], i, b(k)));
        }
    }

    SparseMatrix<double> B = SparseMatrix<double> (N,N);
    B.setFromTriplets(B_triplets.begin(), B_triplets.end());
    B.makeCompressed();

    return B;
}


int main(int argc, char **argv) {
    int N = 100;
    if (argc > 1) {
        N = std::stoi(argv[1]);
    }

    srand(time(nullptr));

    // run test with small matrices
    std::cout << "--> Small Test" << std::endl;
    int n = 5;
    SparseMatrix<double> A(n,n);
    A.coeffRef(3,4) = 1;
    A.coeffRef(4,3) = 2;
    A.coeffRef(1,4) = 3;
    A.coeffRef(3,3) = 4;
    A.coeffRef(3,2) = 4;
    A.coeffRef(2,3) = 4;
    A.coeffRef(2,2) = 5;
    A.coeffRef(3,1) = 6;
    A.coeffRef(0,0) = 9;

    SparseMatrix<double> B = spai(A);
    SparseMatrix<double> I_Small(n,n);
    I_Small.setIdentity();

    std::cout << "A = " << std::endl << A << std::endl;
    std::cout << "B = " << std::endl << B << std::endl;
    std::cout << "Error: " << (I_Small - A*B).norm() << std::endl;
    std::cout << std::endl;

    // run test with large matrices
    std::cout << "--> Big Test (n = " << N*N << ")" << std::endl;
    SparseMatrix<double> C(N*N,N*N);

    SparseMatrix<double> I_Big(N,N);
    I_Big.setIdentity();
    MatrixXd R = MatrixXd::Random(N,N);
    C = kroneckerProduct(R, I_Big);

    Timer tm;
    tm.start();
    SparseMatrix<double> D = spai(C);
    tm.stop();

    SparseMatrix<double> I_Bigbig(N*N,N*N);
    I_Bigbig.setIdentity();

    std::cout << "Error: "  << (I_Bigbig - C*D).norm() << std::endl
              << "Elapsed:" << tm.duration() << " s"  << std::endl;

    return 0;
}

