#include <iostream>
#include <vector>
#include <Eigen/Dense>

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

//! \brief Golub-Welsh implementation 5.3.35
//! \param[in] n number of Gauss nodes
//! \param[out] w weights for interval [-1,1]
//! \param[out] xi ordered nodes for interval [-1,1]
void gaussrule(int n, Vector & w, Vector & xi) {
    assert(n > 0 && "n must be positive!");

    w.resize(n);
    xi.resize(n);
    if( n == 0 ) {
        xi(0) = 0;
        w(0) = 2;
    } else {
//         Vector b(n-1);
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n,n);

        for(int i = 1; i < n; ++i) {
            double d = (i) / sqrt(4. * i * i - 1.);
            J(i,i-1) = d;
            J(i-1,i) = d;
        }

        Eigen::EigenSolver<Eigen::MatrixXd> eig(J);

        xi = eig.eigenvalues().real();
        w = 2 * eig.eigenvectors().real().topRows<1>().cwiseProduct(eig.eigenvectors().real().topRows<1>());
    }

    std::vector<std::pair<double,double>> P;
    P.reserve(n);
    for(int i = 0; i < n; ++i) {
        P.push_back(std::pair<double,double>(xi(i),w(i)));
    }
    std::sort(P.begin(), P.end());
    for(int i = 0; i < n; ++i) {
        xi(i) = std::get<0>(P[i]);
        w(i) = std::get<1>(P[i]);
    }
}

//! \brief Compute the function g in the Gauss nodes
//! \param[in] f object with an evaluation operator (e.g. a lambda function) representing the function f
//! \param[in] n number of nodes
//! \param[out] Vector containing the function g calculated in the Gauss nodes.
template<typename Function>
Vector comp_g_gausspts(Function f, int n) {

    Vector g(n);

    // TODO: problem 4b

    return g;
}


int main() {

    //// PROBLEM 4c
    // TODO: problem 4c, Test the implementation by calculating g(xi^21_10) for f(y)=exp(-|0.5-y|)

    int n = 21;
    auto f = [] (double y) {
        return exp(-std::abs(.5-y));
    };
    Vector g = comp_g_gausspts(f,n);
    std::cout << "*** PROBLEM 4c:" << std::endl;

    std::cout << "g(xi^" << n << "_" << (n+1)/2 << ") = " << g((n+1)/2-1)
              << std::endl;

    return 0;
}

