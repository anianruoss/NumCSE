#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>


using namespace Eigen;

class MatrixCOO {
  public:
    std::vector<Triplet<double>> triplets;

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
};

class MatrixCRS {
  public:
    std::vector<double> val;
    std::vector<int> col_ind;
    std::vector<int> row_ptr = {0};

    MatrixCRS(MatrixXd &A) {
        for (int i = 0; i < A.rows(); ++i) {
            for (int j = 0; j < A.cols(); ++j) {
                if (A(i,j) != 0) {
                    val.push_back(A(i,j));
                    col_ind.push_back(j);
                }
            }

            if (row_ptr.size()) {
                row_ptr.push_back(col_ind.size());
            }
        }
    }

	// Complexity: O(t*log(t))	(for t = #triplets)
    MatrixCRS(MatrixCOO &A) {
        A.sort_byRow();
        std::vector<Triplet<double>> &triplets = A.triplets;

        for (unsigned int i = 0; i < triplets.size(); ++i) {
            val.push_back(triplets[i].value());
            col_ind.push_back(triplets[i].col());

            if (row_ptr.size()-1 < triplets[i].row()) {
                for (int k = row_ptr.size()-1; k < triplets[i].row(); ++k) {
                    row_ptr.push_back(col_ind.size()-1);
                }
            }
        }

        row_ptr.push_back(col_ind.size());
    }

    int rows() {
        return row_ptr.size() - 1;
    }

    int cols() {
        int maxCol = -1;

        for (auto c : col_ind) {
            if (maxCol < c) {
                maxCol = c;
            }
        }

        return ++maxCol;
    }

    MatrixXd toDense() {
        MatrixXd dense(this->rows(), this->cols());
        int k = 0;

        for (int i = 0; i < this->rows(); ++i) {
            for (int j = row_ptr[i]; j < row_ptr[i+1]; ++j) {
                dense(i, col_ind[j]) = val[k++];
            }
        }

        return dense;
    }
};


int main() {
    MatrixXd A(18,17);
    A <<
      2,-1, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      -1, 4,-1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0,
      0,-1, 3,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,
      0, 0,-1, 3, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0,-1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      -1,-1, 0, 0, 4, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0,-1, 0,
      0, 0, 0,-1, 0, 0, 0, 4, 0, 0,-1, 0, 0, 0, 0, 0,-1,
      0, 0, 0, 0,-1, 0, 0, 0, 4, 0, 0,-1,-1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 4,-1, 0, 0,-1,-1, 0,-1,
      0, 0, 0, 0, 0, 0, 0,-1, 0,-1, 3, 0, 0, 0,-1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 2,-1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0,-1, 4,-1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0,-1, 3,-1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0,-1,-1, 0, 0,-1, 3, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,-1, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,-1,
      0, 0,-1,-1, 0, 0, 0,-1, 0,-1, 0, 0, 0, 0, 0,-1, 5;

    std::cout << "--> Converting to COO and back" << std::endl;
    MatrixCOO B(A);
    MatrixXd C = B.toDense();
    std::cout << "Error = " << (A-C).norm() << std::endl;
    std::cout << std::endl;

    std::cout << "--> Converting to CRS and back" << std::endl;
    MatrixCRS D(A);
    MatrixXd E = D.toDense();
    std::cout << "Error = " << (A-E).norm() << std::endl;
    std::cout << std::endl;

    std::cout << "--> Converting from COO to CRS" << std::endl;
    MatrixCRS F(B);
    MatrixXd G = F.toDense();
    std::cout << "Error = " << (A-G).norm() << std::endl;

    return 0;
}

