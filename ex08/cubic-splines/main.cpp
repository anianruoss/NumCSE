#include <eigen3/Eigen/Dense>
#include <mgl2/mgl.h>
#include <iostream>


using namespace Eigen;

// Returns the matrix representing the spline interpolating the data
// with abscissae T and ordinatae Y. Each column represents the coefficients
// of the cubic polynomial on a subinterval.
// Assumes T is sorted, has no repeated elements and T.size() == Y.size().
MatrixXd cubicSpline(const VectorXd &T, const VectorXd &Y) {
    int n = T.size() - 1; // T and Y have length n+1

    MatrixXd spline(4, n);
    spline.row(0) = Y.head(n).transpose();

    VectorXd h(n);

    for (int i = 0; i < n; ++i) {
        h(i) = T(i+1) - T(i);
    }

    MatrixXd triDiag = MatrixXd::Zero(n-1,n-1);
    triDiag(0,0) = (h(0) + h(1))/3;

    for (int i = 1; i < n-1; ++i) {
        triDiag(i,i) = (h(i) + h(i+1))/3;
        triDiag(i,i-1) = h(i)/6;
        triDiag(i-1,i) = h(i)/6;
    }

    VectorXd r(n-1);

    for (int i = 0; i < n-1; ++i) {
        r(i) = (Y(i+2) - Y(i+1))/h(i+1) - (Y(i+1) - Y(i))/h(i);
    }

    VectorXd mu = VectorXd::Zero(n+1);
    mu.head(n).tail(n-1) = triDiag.fullPivLu().solve(r);

    for (int i = 0; i < n; ++i) {
        spline(1,i) = (Y(i+1) - Y(i))/h(i) - h(i)*(2*mu(i) + mu(i+1))/6;
        spline(2,i) = mu(i)/2;
        spline(3,i) = (mu(i+1) - mu(i))/(6*h(i));
    }

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
            if (T(i+1) >= evalT(k)) {
                out(k) += S(0,i) + S(1,i) * (evalT(k) - T(i));
                out(k) += S(2,i) * std::pow(evalT(k) - T(i),2);
                out(k) += S(3,i) * std::pow(evalT(k) - T(i),3);
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

    int len = 1 << 9;
    VectorXd evalT = VectorXd::LinSpaced(len, T(0), T(T.size()-1));

    VectorXd evalSpline = evalCubicSpline(cubicSpline(T, Y), T, evalT);

    std::vector<double> tvec(T.data(), T.data() + T.rows()*T.cols());
    std::vector<double> yvec(Y.data(), Y.data() + Y.rows()*Y.cols());

    double *t = &tvec[0];
    double *y = &yvec[0];

    mglData refx, refy;
    refx.Link(t, 9);
    refy.Link(y, 9);

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
    gr.WriteFrame("spline.eps");

    return 0;
}

