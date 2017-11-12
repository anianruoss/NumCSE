#ifndef _FFT_LOCAL_
#define _FFT_LOCAL_

# include <Eigen/Dense>
# include <unsupported/Eigen/FFT>

/*!
 * \brief fft Two-dimensional real DFT for matrices.
 * Performs 2-dimensional real DFT, using one-dimensional
 * real/complex DFT.
 * \param X A real matrix.
 * \return A complex matrix, with Fourier coeffficients of X.
 */
Eigen::MatrixXcd fft2r(const Eigen::MatrixXd& X) {
    
	const long m = X.rows(), n = X.cols();
		
	Eigen::FFT<double> fft;
	Eigen::MatrixXcd Y1(m, n);
	Eigen::MatrixXcd Y2(n, m);

	for (long j = 0; j < n; ++j) {
		Eigen::VectorXd Xj = X.col(j);
		Y1.col(j) = fft.fwd(Xj);
	}
    
	for (long j = 0; j < m; ++j) {
		Eigen::VectorXcd Yj = Y1.row(j);
		Y2.col(j) = fft.fwd(Yj);
	}
    
	return Y2.transpose();
}

#endif
