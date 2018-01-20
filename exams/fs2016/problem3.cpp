#include <iostream>
#include <cassert>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <mgl2/mgl.h>

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

// Multiple point evaluation of Hermite polynomial
// \Blue{$y_1$}, \Blue{$y_2$}: data values
// \Blue{$c_1$}, \Blue{$c_2$}: slopes
Vector hermloceval(Vector t, double t1, double t2, double y1, double y2,
                   double c1, double c2) {
    const double h = t2-t1,a1 = y2-y1, a2 = a1-h*c1, a3 = h*c2-a1-a2;
    t = ((t.array()-t1)/h).matrix();

    return (y1+(a1+(a2+a3*t.array())*(t.array()-1))*t.array()).matrix();
}

//! \brief Computes the slopes for natural cubic spline interpolation
//! \param[in] t vector of nodes
//! \param[in] y vector of values at the nodes
//! \param[out] c vector of slopes at the nodes
void natsplineslopes (const Vector &t, const Vector &y, Vector &c) {
    // Size check
    assert( ( t.size() == y.size() ) && "Error: mismatched size of t and y!");

    // n+1 is the number of conditions (t goes from t_0 to t_n)
    int n = t.size() - 1;

    Vector h(n);
    // Vector containing increments (from the right)
    for(int i = 0; i < n; ++i) {
        h(i) = t(i+1) - t(i);
        // Check that t is sorted
        assert( ( h(i) > 0 ) && "Error: array t must be sorted!");
    }

    // System matrix and rhs as in (3.5.9), we remove first and last row
    // (known by natural contition)
    Eigen::SparseMatrix<double> A(n+1,n+1);
    Vector b(n+1);

    // WARNING: sparse reserve space
    A.reserve(3);

    // Fill in natural conditions (3.5.10) for matrix
    A.coeffRef(0,0) = 2./h(0);
    A.coeffRef(0,1) = 1./h(0);
    A.coeffRef(n,n-1) = 1./h(n-1);
    A.coeffRef(n,n) = 2./h(n-1);

    // Reuse computation for rhs
    double bold = (y(1) - y(0)) / (h(0)*h(0));
    b(0) = 3.*bold; // Fill in natural conditions (3.5.10)
    // Fill matrix A and rhs b
    for(int i = 1; i < n; ++i) {
        // Fill in a
        A.coeffRef(i,i-1) = 1./h(i-1);
        A.coeffRef(i,i) = 2./h(i-1) + 2./h(i);
        A.coeffRef(i,i+1) = 1./h(i);

        // Reuse computation for rhs b
        double bnew = (y(i+1) - y(i)) / (h(i)*h(i));
        b(i) = 3. * (bnew + bold);
        bold = bnew;
    }
    b(n) = 3.*bold; // Fill in natural conditions (3.5.10)
    // Compress the matrix
    A.makeCompressed();

    // Factorize A and solve system A*c(1:end) = b
    Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;
    lu.compute(A);
    c = lu.solve(b);
}

//! \brief Computes the slopes for not-a-knot cubic spline interpolation
//! \param[in] t vector of nodes
//! \param[in] y vector of values at the nodes
//! \param[out] c vector of slopes at the nodes
void notknotsplines (const Vector &t, const Vector &y, Vector &c) {
    // Size check
    assert( ( t.size() == y.size() ) && "Error: mismatched size of t and y!");

    // n+1 is the number of conditions (t goes from t_0 to t_n)
    int n = t.size() - 1;

    Vector h(n);
    // Vector containing increments (from the right)
    for(int i = 0; i < n; ++i) {
        h(i) = t(i+1) - t(i);
        // Check that t is sorted
        assert( ( h(i) > 0 ) && "Error: array t must be sorted!");
    }

    // System matrix and rhs as in (3.5.9), we remove first and last row
    // (known by natural contition)
    Eigen::SparseMatrix<double> A(n+1,n+1);
    A.reserve(3*(n+1));
    Vector b(n+1);

    // Fill in not-a-knot conditions for matrix
    A.coeffRef(0,0) = 1. / (h(0)*h(0));
    A.coeffRef(0,1) = 1. / (h(0)*h(0)) - 1. / (h(1)*h(1));
    A.coeffRef(0,2) = - 1. / (h(1)*h(1));
    A.coeffRef(n,n-2) = 1. / (h(n-2)*h(n-2));
    A.coeffRef(n,n-1) = 1. / (h(n-2)*h(n-2)) - 1. / (h(n-1)*h(n-1));
    A.coeffRef(n,n) = - 1. / (h(n-1)*h(n-1));

    // Fill matrix A and rhs b
    // Fill in not-a-knot conditions
    b(0) = 2.*((y(1)-y(0))/std::pow(h(0),3) + (y(1)-y(2))/std::pow(h(1),3));
    b(n) = 2.*(y(n-1)-y(n-2))/std::pow(h(n-2),3);
    b(n) += 2.*(y(n-1)-y(n))/std::pow(h(n-1),3);

    // Reuse computation for rhs
    double bold = (y(1) - y(0)) / (h(0)*h(0));

    for(int i = 1; i < n; ++i) {
        // Fill in a
        A.coeffRef(i,i-1) = 1./h(i-1);
        A.coeffRef(i,i) = 2./h(i-1) + 2./h(i);
        A.coeffRef(i,i+1) = 1./h(i);

        // Reuse computation for rhs b
        double bnew = (y(i+1) - y(i)) / (h(i)*h(i));
        b(i) = 3. * (bnew + bold);
        bold = bnew;
    }

    // Compress the matrix
    A.makeCompressed();

    // Factorize A and solve system A*c(1:end) = b
    Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;
    lu.compute(A);
    c = lu.solve(b);
}


int main() {
    const size_t n = 15;
    auto f = [] (double x) {
        return std::sin(x*4.*M_PI);
    };

    Vector t = Vector::Random(n);
    std::sort(t.data(), t.data()+t.size(), [] (double a, double b) {
        return a < b;
    });

    Vector y = t.unaryExpr(f);

    Vector tplot = Vector::LinSpaced(500, t(0), t(n-1));
    Vector yplot = tplot.unaryExpr(f);

    Vector c_nat = Vector::Zero(n);
    Vector c_knot = Vector::Zero(n);

    natsplineslopes(t, y, c_nat);
    notknotsplines(t, y, c_knot);

    std::vector<double> t_nat, y_nat;

    for (size_t i = 0; i < n-1; ++i) {
        Vector vx = Vector::LinSpaced(100, t(i), t(i+1));
        Vector px =	hermloceval(vx,t(i),t(i+1),y(i),y(i+1),c_nat(i),c_nat(i+1));
        std::copy(vx.data(), vx.data()+vx.size(), std::back_inserter(t_nat));
        std::copy(px.data(), px.data()+px.size(), std::back_inserter(y_nat));
    }

    std::vector<double> t_knot, y_knot;

    for (size_t i = 0; i < n-1; ++i) {
        Vector vx = Vector::LinSpaced(100, t(i), t(i+1));
        Vector px =	hermloceval(vx,t(i),t(i+1),y(i),y(i+1),c_knot(i),c_knot(i+1));
        std::copy(vx.data(), vx.data()+vx.size(), std::back_inserter(t_knot));
        std::copy(px.data(), px.data()+px.size(), std::back_inserter(y_knot));
    }

    mglData ptsx, ptsy;
    ptsx.Link(t.data(), t.size());
    ptsy.Link(y.data(), y.size());

    mglData refx, refy;
    refx.Link(tplot.data(), tplot.size());
    refy.Link(yplot.data(), yplot.size());

    mglData natx, naty;
    natx.Link(&t_nat[0], t_nat.size());
    naty.Link(&y_nat[0], y_nat.size());

    mglData knotx, knoty;
    knotx.Link(&t_knot[0], t_knot.size());
    knoty.Link(&y_knot[0], y_knot.size());

    mglGraph gr;
    gr.Title("Natural vs. Not-a-Knot Splines");
    gr.SetRanges(t(0), t(n-1), -2.5, 1.5);
    gr.Plot(ptsx, ptsy, "r *");
    gr.AddLegend("Reference Points", "r *");
    gr.Plot(refx, refy, "b");
    gr.AddLegend("sin(4*PI*x)", "b");
    gr.Plot(natx, naty, "c");
    gr.AddLegend("Natural", "c");
    gr.Plot(knotx, knoty, "g");
    gr.AddLegend("Not-a-Knot", "g");
    gr.Legend(0);
    gr.Axis();
    gr.WriteFrame("cubicSplineInterpolations.eps");

    return 0;
}

