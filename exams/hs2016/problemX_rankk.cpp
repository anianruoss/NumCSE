#include <cmath>
#include <iostream>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <unsupported/Eigen/FFT>

using namespace Eigen;

/*! @brief Build the matrix $\mathbf{C}$ corresponding to the mapping $l$.
 *! @param[in] h  Vector of entries of $\VC$.
 *! @return C The matrix $\mathbf{C}$ representing the linear mapping.
 */
MatrixXd buildLTFIRMatrix(const VectorXd & h) {
    const size_t n = h.size();
    MatrixXd C = MatrixXd::Zero(2*n-1, n);

    for (size_t i = 0; i < n; ++i) {
        C.col(i).tail(2*n-1-i).head(n) = h;
    }

    return C;
}

/*!
 * \brief The LTFIR_lowrank class evaluates a LTFIR filter
 * using a low rank approximation.
 */
class LTFIR_lowrank {
  public:
    /*! @brief Constructor of rank-$k$ approximation LTFIR_lowrank.
     *! @param[in]  h Vector of entries of matrix of the filter.
     *! @param[in]  k Rank of the approximation.
     */
    LTFIR_lowrank(const VectorXd& h, unsigned int k) {
        MatrixXd C = buildLTFIRMatrix(h);

        // complexity: O(n^3)
        JacobiSVD<MatrixXd> svd(C, ComputeThinU | ComputeThinV);
        MatrixXd Sigma = svd.singularValues().head(k).asDiagonal();

        // complexity: O(k) (optimised product with diagonal matrix)
        U_ = svd.matrixU().leftCols(k) * Sigma;
        Vt_ = svd.matrixV().leftCols(k).transpose();
    }

    /*! @brief Operator evaluating $\tilde{\mathbf{C}}\mathbf{x}$
     *! @param[in] x Input vector.
     *! @return      Output vector $y = \tilde{\mathbf{C}}\mathbf{x}$
     */
    VectorXd operator()(const VectorXd& x) const {
        // complexity: O(nk)
        VectorXd tmp = Vt_ * x;
        return U_ * tmp;
    }

  private:
    MatrixXd U_;
    MatrixXd Vt_;
};

/*!
 * \brief The LTFIR_freq class evaluates a LTFIR filter
 * using frequency filtering.
 */
class LTFIR_freq {
  public:
    /*!
     *! @brief Constructor of FFT-based frequency filter LTFIR_freq.
     *! @param[in]  h  Vector of entries for the matrix of the filter.
     *! @param[in]  k  Maximum number of frequencies (low-pass filter).
     */
    LTFIR_freq(const VectorXd& h, unsigned k) {
        n_ = h.size();
        k_ = k;

        VectorXd h_ = h;
        h_.conservativeResizeLike(VectorXd::Zero(2*n_-1));

        // Forward DFT
        FFT<double> fft;
        ch_ = fft.fwd(h_);
    }

    /*!
     *! @brief Low-pass filtering operator.
     *! @param[in]  x  Input vector (to be filtered).
     *! @return        Filtered output vector.
     */
    VectorXd operator()(const VectorXd& x) const {
        assert(x.size() == n_ && "x must have same length of h");

        VectorXd x_ = x;
        x_.conservativeResizeLike(VectorXd::Zero(2*n_-1));
        // Forward DFT
        FFT<double> fft;
        VectorXcd cx = fft.fwd(x_);
        VectorXcd c  = ch_.cwiseProduct(cx);
        // Set high frequency coefficients to zero
        VectorXcd clow = c;
        for(int j=-k_; j<=+k_; ++j) clow(n_+j) = 0;
        // Inverse DFT
        return fft.inv(clow).real();
    }

  private:

    int n_;
    int k_;
    VectorXcd ch_;
};


int main() {
    unsigned int n = 6;
    unsigned int k = 4;

    // PART 1: build system matrix C
    std::cout << "**** PART 1 ****" << std::endl;

    VectorXd h = VectorXd::LinSpaced(n,1,n);
    std::cout << "h = " << h.transpose() << std::endl << std::endl;

    MatrixXd C = buildLTFIRMatrix(h);
    MatrixXd C_exact(2*n-1, n);
    C_exact <<
            1, 0, 0, 0, 0, 0,
            2, 1, 0, 0, 0, 0,
            3, 2, 1, 0, 0, 0,
            4, 3, 2, 1, 0, 0,
            5, 4, 3, 2, 1, 0,
            6, 5, 4, 3, 2, 1,
            0, 6, 5, 4, 3, 2,
            0, 0, 6, 5, 4, 3,
            0, 0, 0, 6, 5, 4,
            0, 0, 0, 0, 6, 5,
            0, 0, 0, 0, 0, 6;

    std::cout << "Matrix C is:" << std::endl
              << C << std::endl << std::endl
              << "Error: " << (C - C_exact).norm() << std::endl << std::endl;

    // PART 2: filter vector x (rank-k filter)
    std::cout << "**** PART 2 ****" << std::endl;

    VectorXd x = VectorXd::Ones(n);;
    std::cout << "x = " << x.transpose() << std::endl << std::endl;

    LTFIR_lowrank H(h,k);
    VectorXd y_H = H(x);
    VectorXd y_exact(2*n-1);
    y_exact <<
            0.980805,
            2.99719,
            5.99522,
            9.97489,
            14.9901,
            20.9862,
            20.1166,
            17.8828,
            14.991,
            11.1175,
            5.88483;

    std::cout << "Vector y is:" << std::endl << y_H << std::endl << std::endl
              << "Error: " << (y_H - y_exact).norm() << std::endl << std::endl;

    // PART 3: filter vector x (FFT-based filter)
    std::cout << "**** PART 3 ****" << std::endl;

    LTFIR_freq F(h,k);
    VectorXd y_F = F(x);
    std::cout << "y_F:" << std::endl << y_F << std::endl;

    return 0;
}

