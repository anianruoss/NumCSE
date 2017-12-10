#pragma once

#include <eigen3/Eigen/Dense>

//! @brief Golub-Welsh implementation 5.3.35
//! @param[in] n number of Gauss nodes
//! @param[out] w weights
//! @param[out] x nodes for interval $[-1,1]$
template <class vector>
inline void golubwelsh(const int n, vector& w, vector& x) {
    w.resize(n);
    x.resize(n);
    if(n == 0) {
        x(0) = 0;
        w(0) = 2;
    } else {
        vector b(n-1);
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n,n);

        for(int i = 1; i < n; ++i) {
            double d = (i) / sqrt(4. * i * i - 1.);
            J(i,i-1) = d;
            J(i-1,i) = d;
        }

        Eigen::EigenSolver<Eigen::MatrixXd> eig(J);

        x = eig.eigenvalues().real();
        vector tmp = eig.eigenvectors().real().topRows<1>();
        w = 2 * tmp.cwiseProduct(tmp);
    }
}

