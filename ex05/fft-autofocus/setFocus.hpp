#pragma once

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include "pgm.hpp" // Contains PGMObject

Eigen::MatrixXd conv2(const Eigen::MatrixXd& P, const Eigen::MatrixXd& S) {
    
	const int m = P.rows(), n = P.cols(), M = S.rows(), N = S.cols();
	Eigen::MatrixXd C(m,n);

	int _tj = N/2 + 1;
	int _ti = M/2 + 1;
	for (int j = 0; j < n; ++j) {
		int _lj = std::max(0, _tj + j - n);
		int _rj = std::min(N, _tj + j);
		for (int i = 0; i < m; ++i) {

			double s = 0;
			int _li = std::max(0, i + _ti - m);
			int _ri = std::min(M, i + _ti);
			for (int k = _li; k < _ri; ++k) {
				for (int l = _lj; l < _rj; ++l) {
					int _i = i - k + M/2;
					int _j = j - l + N/2;
					s += P(_i, _j) * S(k, l);
				}
			}

			C(i, j) = s;
		}
	}

	return C;
}


Eigen::MatrixXd set_focus(double f) {

	std::ifstream file("perfectImage.pgm");
	PGMObject p;
	file >> p;

	Eigen::MatrixXd M;
	p.get_data(M);

	double f0 = (1 << 1);
	double ep = std::max(std::abs(f - f0),
                         std::numeric_limits<double>::epsilon());

	unsigned int s = 16;
	Eigen::MatrixXd S(s,s);
	for(unsigned int i = 0; i < s; ++i) {
		for(unsigned int j = 0; j < s; ++j) {
			S(i,j) = 1. / (1 + (i*i + j*j)/ep);
		}
	}
	S /= S.sum();

	return conv2(M, S);
}
