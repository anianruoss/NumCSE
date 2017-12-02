#include <eigen3/Eigen/Dense>


using namespace Eigen;

// IN: t = node set (mutually different)
// y = nodal values
// OUT: coeffs = coefficients of polynomial in Newton basis
VectorXd divDiff(const VectorXd& t, const VectorXd& y) {

    // TODO

}

// Evaluation of polynomial in Newton basis (divided differences)
// IN: t = nodes (mutually different)
// y = values in t
// x = evaluation points (as Eigen::Vector)
// OUT: p = values in x */
void intPolyEval (const VectorXd& t, const VectorXd& y, const VectorXd& x, VectorXd& p) {

    const unsigned n = y.size()-1 ;
    // get Newton coefficients of polynomial (non in-situ implementation!)
    VectorXd coeffs;
    coeffs = divDiff(t, y);

    // evaluate
    VectorXd ones = VectorXd::Ones(x.size());
    p = coeffs(n)*ones;

    for (int j=n-1; j>= 0; --j)
        p = (x - t(j)*ones).cwiseProduct(p) + coeffs(j)*ones;
}

