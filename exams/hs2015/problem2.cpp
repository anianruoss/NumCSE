#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

using Triplet = Eigen::Triplet<double>;
using Triplets = std::vector<Triplet>;
using Vector = Eigen::VectorXd;
using index_t = std::ptrdiff_t;

//! \brief Class representing a sparse matrix in Ellpack format
class EllpackMat {
  public:
    EllpackMat(const Triplets & triplets, index_t m, index_t n);

    double operator() (index_t i, index_t j) const;

    void mvmult(const Vector &x, Vector &y) const;

    void mtvmult(const Vector &x, Vector &y) const;
  private:
    std::vector<double> val; //!< Vector containing value corresponding to entry in col
    std::vector<index_t> col; //! Vector containing columns, partitioned into same-size rows

    index_t maxcols; // Max elements per row
    index_t m,n; // Number of rows, number of columns
};

//! \brief Construct  a matrix Ellpack from triplet format
//! \param[in] triplets vector of Eigen::Triplets
//! \param[in] m rows of matrix
//! \param[in] n columns of matrix
//! \param[in] maxcols max. number of nonzeros of the matrix
EllpackMat::EllpackMat(const Triplets & triplets,
                       index_t m, index_t n) : m(m), n(n) {
    std::vector<unsigned int> counters(m);
    std::fill(counters.begin(), counters.end(), 0);

    maxcols = 0;

    for	(const auto tr : triplets) {
        if (++counters[tr.row()] > maxcols) {
            maxcols = counters[tr.row()];
        }
    }

    col.resize(m*maxcols, -1);
    val.resize(m*maxcols, 0);

    for (const auto tr : triplets) {
        assert(0 <= tr.row() && tr.row() < m && 0 <= tr.col() && tr.col() < n &&
               "Index out of bounds!");

        index_t I;
        for (I = tr.row()*maxcols; I < (tr.row()+1)*maxcols; ++I) {
            if (col[I] == tr.col()) {
                val[I] += tr.value();
                break;
            } else if (col[I] == -1) {
                col[I] = tr.col();
                val[I] = tr.value();
                break;
            }
        }

        assert(I < (tr.row()+1)*maxcols && "Not enough colums reserved!");
    }
}

//! \brief Retrieve value of entry at index (i,j)
//! \param[in] i row index i
//! \param[in] j column index j
//! \return value at (i,j)
double EllpackMat::operator() (index_t i, index_t j) const {
    assert(0 <= i && i < m && 0 <= j && j < n
           && "Index out of bounds!");

    for(index_t l = i*maxcols; l < (i+1)*maxcols; ++l) {
        if( col[l] == j ) return val[l];
    }
    return 0.;
}

//! \brief Computes (*this)*x
//! \param[in] x vector x for mat-vec mult
//! \param[out] y result y = (*this)*x
void EllpackMat::mvmult(const Vector &x, Vector &y) const {
    assert(x.size() == n && "Incompatible vector x size!");
    assert(y.size() == m && "Incompatible vector y size!");

    y = Vector::Zero(m);

    for (index_t i = 0; i < m; ++i) {
        for (index_t l = i*maxcols; l < (i+1)*maxcols; ++l) {
            if (col[l] == -1) break;
            y(i) += x(col[l]) * val[l];
        }
    }
}

//! \brief Computes (*this)'*x, A' is transposed
//! \param[in] x vector x for mat-vec mult
//! \param[out] y result y = (*this)'*x
void EllpackMat::mtvmult(const Vector &x, Vector &y) const {
    assert(x.size() == m && "Incompatible vector x size!");
    assert(y.size() == n && "Incompatible vector y size!");

    y = Vector::Zero(n);

    for (index_t i = 0; i < m; ++i) {
        for (index_t l = i*maxcols; l < (i+1)*maxcols; ++l) {
            if (col[l] == -1) break;
            y(col[l]) += x(i) * val[l];
        }
    }
}


int main(int, char**) {
    // Vector of triplets
    Triplets triplets;

    // Data
    unsigned int m = 3, n = 6;
    unsigned int ntriplets = 9;

    // Reserve space for triplets
    triplets.reserve(ntriplets);

    // Fill in some triplet
    triplets.push_back(Triplet(1,2,4));
    triplets.push_back(Triplet(0,0,5));
    triplets.push_back(Triplet(1,2,6));
    triplets.push_back(Triplet(2,5,7));
    triplets.push_back(Triplet(0,4,8));
    triplets.push_back(Triplet(1,3,9));
    triplets.push_back(Triplet(2,2,10));
    triplets.push_back(Triplet(2,1,11));
    triplets.push_back(Triplet(1,0,12));

    // Build matrix (Eigen Sparse)
    Eigen::SparseMatrix<double> S(m,n);
    S.setFromTriplets(triplets.begin(), triplets.end());

    // Build Ellpack matrix
    EllpackMat E(triplets, m, n);

    //// PROBLEM 2 TEST
    std::cout << "*** PROBLEM 2, testing:" << std::endl;
    std::cout << " ------------- Test of y = A^t*x ------------- " << std::endl;
    Eigen::VectorXd x(6);
    x << 4,5,6,7,8,9;
    Eigen::VectorXd Ex = Eigen::VectorXd::Zero(m);
    E.mvmult(x, Ex);
    std::cout << "Sparse S*x  = " << (S*x).transpose() << std::endl;
    std::cout << "Ellpack E*x = " << Ex.transpose() << std::endl << std::endl;
    std::cout << " ------------- Test of x = A*y ------------- " << std::endl;
    Eigen::VectorXd y(3);
    y << 1,2,3;
    Eigen::VectorXd Ey = Eigen::VectorXd::Zero(n);
    E.mtvmult(y, Ey);
    std::cout << "Sparse S^T*y  = " << (S.transpose()*y).transpose()
              << std::endl;
    std::cout << "Ellpack E^T*y = " << Ey.transpose() << std::endl;

    return 0;
}

