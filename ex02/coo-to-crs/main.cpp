#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

#include <iostream>
#include <vector>


using namespace Eigen;

class MatrixCOO {
    friend class MatrixCRS;

  public:
    MatrixCOO(MatrixXd &A) : rows(A.rows()), cols(A.cols()) {
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
        [] (Triplet<double> a, Triplet<double> b) {
            return a.row() < b.row();
        });
    }

    MatrixXd toDense() {
        MatrixXd A = MatrixXd::Zero(rows, cols);

        for (auto t : triplets) {
            A(t.row(), t.col()) += t.value();
        }

        return A;
    }

  private:
    int rows;
    int cols;
    std::vector<Triplet<double>> triplets;
};

class MatrixCRS {
  public:
    MatrixCRS(MatrixXd &A) {
        for (int i = 0; i < A.rows(); ++i) {
            row_ptr.push_back(val.size());

            for (int j = 0; j < A.cols(); ++j) {
                if (A(i,j) != 0) {
                    val.push_back(A(i,j));
                    col_ind.push_back(j);
                }
            }
        }

        row_ptr.push_back(val.size()+1);
    }

    // Complexity: O(t*log(t))	(for t = #triplets)
    MatrixCRS(MatrixCOO &A) {
        A.sort_byRow();

        for (auto it = A.triplets.begin(); it != A.triplets.end(); ++it) {
            if (it->row()+1 > row_ptr.size()) {
                if (it->row() > (it-1)->row()+1) {
                    int empty_rows = it->row() - (it-1)->row()-1;
                    for (int i = 0; i < empty_rows; ++i) {
                        row_ptr.push_back(val.size());
                    }
                }
                row_ptr.push_back(val.size());
            }
            val.push_back(it->value());
            col_ind.push_back(it->col());
        }

        row_ptr.push_back(val.size()+1);
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
        const int n = row_ptr.size();
        MatrixXd A = MatrixXd::Zero(rows(), cols());

        for (int i = 0; i < n-2; ++i) {
            for (int j = row_ptr[i]; j < row_ptr[i+1]; ++j) {
                A(i, col_ind[j]) = val[j];
            }
        }

        for (int j = row_ptr[n-2]; j < row_ptr[n-1]-1; ++j) {
            A(rows()-1, col_ind[j]) = val[j];
        }

        return A;
    }

  private:
    std::vector<double> val;
    std::vector<int> col_ind;
    std::vector<int> row_ptr;
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

