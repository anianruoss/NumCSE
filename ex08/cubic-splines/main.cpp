#include <eigen3/Eigen/Dense>
#include <mgl2/mgl.h>

#include <iostream>

using namespace Eigen;

// Returns the matrix representing the spline interpolating the data
// with abscissae T and ordinatae Y. Each column represents the coefficients
// of the cubic polynomial on a subinterval.
// Assumes T is sorted, has no repeated elements and T.size() == Y.size().
MatrixXd cubicSpline(const VectorXd &T, const VectorXd &Y) {
    const int n = T.size() - 1;

    VectorXd h = T.tail(n) - T.head(n);
    VectorXd slope = (Y.tail(n) - Y.head(n)).cwiseQuotient(h);
    VectorXd r = slope.tail(n-1) - slope.head(n-1);

    MatrixXd triDiag = MatrixXd::Zero(n-1, n-1);
    triDiag.diagonal() = (h.tail(n-1) + h.head(n-1)) / 3.;
    triDiag.diagonal(1) = triDiag.diagonal(-1) = h.head(n-1).tail(n-2) / 6.;

    VectorXd sigma = VectorXd::Zero(n+1);
    sigma.head(n).tail(n-1) = triDiag.partialPivLu().solve(r);

    MatrixXd spline = MatrixXd::Zero(4, n);
    spline.row(0) = Y.head(n);
    spline.row(1) = slope - h.cwiseProduct(2*sigma.head(n)+sigma.tail(n)) / 6.;
    spline.row(2) = sigma.head(n) / 2.;
    spline.row(3) = (sigma.tail(n) - sigma.head(n)).cwiseQuotient(6. * h);

    return spline;
}

// Returns the values of the spline S calculated in the points X.
// Assumes T is sorted, with no repetetions.
VectorXd evalCubicSpline(const MatrixXd &S, const VectorXd &T,
                         const VectorXd &evalT) {
    const int n = evalT.size();
    VectorXd out = VectorXd::Zero(n);

    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < S.cols(); ++i) {
            if (T(i) <= evalT(k) && evalT(k) <= T(i+1)) {
                double x = evalT(k) - T(i);
                out(k) = S(0,i) + x*(S(1,i) + x*(S(2,i) + x*S(3,i)));
                break;
            }
        }
    }

    return out;
}

int main() {
    VectorXd T(9);
    VectorXd Y(9);
    T << 0, 0.4802, 0.7634, 1, 1.232, 1.407, 1.585, 1.879, 2;
    Y << 0., 0.338, 0.7456, 0, -1.234, 0 , 1.62, -2.123, 0;

    const int len = 1 << 9;
    VectorXd evalT = VectorXd::LinSpaced(len, T(0), T(T.size()-1));

    VectorXd evalSpline = evalCubicSpline(cubicSpline(T, Y), T, evalT);

    mglData refx, refy;
    refx.Link(T.data(), T.size());
    refy.Link(Y.data(), Y.size());

    mglData datx, daty;
    datx.Link(evalT.data(), len);
    daty.Link(evalSpline.data(), len);

    mglGraph gr;
    gr.SetRanges(0, 2, -3, 3);
    gr.Plot(refx, refy, "g *");
    gr.AddLegend("Reference Points", "g *");
    gr.Plot(datx, daty, "b");
    gr.AddLegend("Splines", "b");
    gr.Legend(2);
    gr.WriteFrame("plots/spline.eps");

    return 0;
}

