#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/Sparse>


using namespace Eigen;

class MatrixCOO {
  public:
    std::vector<Triplet<double>> triplets;

    MatrixCOO() {}

    MatrixCOO(std::vector<Triplet<double>> &triplets_) {
        triplets = triplets_;
    }

    MatrixCOO(MatrixXd &A) {
        for (int i = 0; i < A.rows(); ++i) {
            for (int j = 0; j < A.cols(); ++j) {
                if (A(i,j) != 0) {
                    Triplet<double> triplet(i,j,A(i,j));
                    triplets.push_back(triplet);
                }
            }
        }
    }

    void sort_byRow() {
        std::sort(triplets.begin(), triplets.end(),
        [] (auto t1, auto t2) {
            return t1.row() < t2.row();
        });
    }

    void sort_byCol() {
        std::sort(triplets.begin(), triplets.end(),
        [] (auto t1, auto t2) {
            return t1.col() < t2.col();
        });
    }

    int cols() {
        int maxCol = -1;

        for (auto t : triplets) {
            if (maxCol < t.col()) {
                maxCol = t.col();
            }
        }

        return ++maxCol;
    }

    int rows() {
        int maxRow = -1;

        for (auto t : triplets) {
            if (maxRow < t.row()) {
                maxRow = t.row();
            }
        }

        return ++maxRow;
    }

    MatrixXd toDense() {
        MatrixXd dense(this->rows(), this->cols());

        for (auto t : triplets) {
            dense(t.row(),t.col()) += t.value();
        }

        return dense;
    }

	// Complexity: O(tA*tB)		(for tI = #triplets of Matrix I)
    static MatrixCOO mult_naive (MatrixCOO &A, MatrixCOO &B) {
        MatrixCOO result;

        for (auto t1 : A.triplets) {
            for (auto t2 : B.triplets) {
                if (t1.col() == t2.row()) {
                    Triplet<double> t(t1.row(), t2.col(), t1.value()*t2.value());
                    result.triplets.push_back(t);
                }
            }
        }

        return result;
    }

	// Complexity: O(n*log(n))		(for n = nnz(A) = nnz(b))
    static MatrixCOO mult_efficient (MatrixCOO &A1, MatrixCOO &A2) {
        // avoid complications when A1 and A2 are same object, copy A2 by value
        if (&A1 == &A2) {
            MatrixCOO copyA2(A2.triplets);
            return mult_efficient(A1, copyA2);
        }

        MatrixCOO result;

		std::vector<Triplet<double>> &l1 = A1.triplets;
		std::vector<Triplet<double>> &l2 = A2.triplets;
		std::vector<Triplet<double>> &lr = result.triplets;

        A1.sort_byCol();
        A2.sort_byRow();

        // build vectors of indices b1 and b2, which contain the indices
        // of elements which begin a new column for A1 and which begin a
        // new row for A2.
		std::vector<int> b1 = {0};
		std::vector<int> b2 = {0};

        for (int i = 0; i < l1.size()-1; i++) {
            if (l1[i].col() != l1[i+1].col()) {
                b1.push_back(i+1);
            }
        }

        b1.push_back(l1.size());

        for (int j = 0; j < l2.size()-1; j++) {
            if (l2[j].row() != l2[j+1].row()) {
                b2.push_back(j+1);
            }
        }

        b2.push_back(l2.size());

        // exploiting the special sorting of l1, l2 we are able to pass
        // in a single loop all and only the elements of the matrices
        // A1, A2 which will be paired when one computes A1*A2.
        int i = 0;
        int j = 0;

        while (i < b1.size()-1 && j < b2.size()-1) {
            if (l1[b1[i]].col() == l2[b2[j]].row()) {
                // in this case, all the associated couples should be
                // multiplied and the resulting triplets
                // have to be added to result.
                int c1, c2;
                for (c1 = b1[i]; c1 < b1[i+1]; c1++) {
                    for (c2 = b2[j]; c2 < b2[j+1]; c2++) {

                        Triplet<double> t(l1[c1].row(), l2[c2].col(),
                                          l1[c1].value() * l2[c2].value());

                        lr.push_back(t);
                    }
                }
                i++;
                j++;
            } else {
                if (l1[b1[i]].col() < l2[b2[j]].row()) {
                    // due to sorting, if the column of l1 is smaller than
                    // the row of l2, eventual couples which can contribute
                    // to the product can be found only by increasing i.
                    i++;
                }
                if (l1[b1[i]].col() > l2[b2[j]].row()) {
                    // same as previous, only the roles are reversed.
                    j++;
                }
            }
        }

        return result;
    }
};


int main() {
    srand(time(nullptr));

    int N = 100;
    double sparseCoeff = 1./std::log(N);

    // generate full N*N matrix with random elements
    MatrixXd A = MatrixXd::Random(N,N);
    MatrixXd B = MatrixXd::Random(N,N);

    // convert to COO format
    MatrixCOO Acoo(A);
    MatrixCOO Bcoo(B);

    // define some random number generator
    std::random_device rd;
    std::mt19937 g(rd());

    // shuffle randomly the elements of the COO triplets
    std::shuffle(Acoo.triplets.begin(), Acoo.triplets.end(), g);
    std::shuffle(Bcoo.triplets.begin(), Bcoo.triplets.end(), g);

    // keep only part of COO triplets, proportionally to sparseCoeff
    int n = sparseCoeff*N*N;
    Acoo.triplets.resize(n);
    Bcoo.triplets.resize(n);
    std::cout << "Multiplication of " << N << "x" << N << " matrices with "
              << n << " non-zero elements." << std::endl;

    // time naive multiplication
    auto start_naive = std::chrono::system_clock::now();
    MatrixCOO::mult_naive(Acoo,Bcoo);
    auto end_naive = std::chrono::system_clock::now();
    double time_naive = (end_naive-start_naive).count();
    std::cout << "Naive: " << time_naive << std::endl;

    // time efficient multiplication
    auto start_eff = std::chrono::system_clock::now();
    MatrixCOO::mult_efficient(Acoo,Bcoo);
    auto end_eff = std::chrono::system_clock::now();
    double time_eff = (end_eff-start_eff).count();
    std::cout << "Efficient: " << time_eff << std::endl;

    std::cout << "Time ratio naive/efficient implementation: "
              << time_naive / time_eff << std::endl;

    return 0;
}

