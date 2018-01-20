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
        // Vector b(n-1);
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n,n);

        for(int i = 1; i < n; ++i) {
            double d = (i) / sqrt(4. * i * i - 1.);
            J(i,i-1) = d;
            J(i-1,i) = d;
        }

        Eigen::EigenSolver<Eigen::MatrixXd> eig(J);

        xi = eig.eigenvalues().real();
        w = 2 * eig.eigenvectors().real().topRows<1>().cwiseProduct(
                eig.eigenvectors().real().topRows<1>());
    }

    std::vector<std::pair<double,double>> P;
    P.reserve(n);
    for (int i = 0; i < n; ++i) {
        P.push_back(std::pair<double,double>(xi(i),w(i)));
    }

    std::sort(P.begin(), P.end());
    for (int i = 0; i < n; ++i) {
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
    Vector w = Vector::Zero(n);
    Vector xi = Vector::Zero(n);

    // Compute Gauss nodes and weights and rescale them to [0,1]
    gaussrule(n, w, xi);
    w /= 2.;
    xi = (xi + Vector::Ones(n)) / 2.;

    Vector p = Vector::Zero(n);
    Vector q = Vector::Zero(n);
    q(n-1) = w(n-1)*std::exp(xi(n-1))*f(xi(n-1));

    for (int i = 1; i < n; ++i) {
        p(i) = p(i-1) + w(i-1)*std::exp(-xi(i-1))*f(xi(i-1));
        q(n-1-i) = q(n-i) + w(n-1-i)*std::exp(xi(n-1-i))*f(xi(n-1-i));
    }

    Vector g = Vector::Zero(n);

    for (int i = 0; i < n; ++i) {
        g(i) = std::exp(xi(i))*p(i) + std::exp(-xi(i))*q(i);
    }

    return g;
}


int main() {
    int n = 21;
    auto f = [] (double y) {
        return exp(-std::abs(.5-y));
    };

    Vector g = comp_g_gausspts(f,n);
    std::cout << "*** PROBLEM 4c:" << std::endl;

    std::cout << "g(xi^" << n << "_" << (n+1)/2 << ")  = " << g((n+1)/2-1)
              << std::endl << "exact result = 1" << std::endl;

    return 0;
}

