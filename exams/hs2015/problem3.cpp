#include <iostream>
#include <vector>

#include <Eigen/Dense>

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

//! \brief Function for the evaluation of sgn(x,y)
double sgn(double x) {
    if (x < 0.) {
        return -1.;
    } else if (x == 0.) {
        return 0.;
    } else {
        return 1.;
    }
}

//! \brief Function for the evaluation of minmod(x,y)
double minmod(double x, double y) {
    if (sgn(x) != sgn(y)) {
        return 0.;
    } else {
        return std::min(std::abs(x), std::abs(y))*sgn(x);
    }
}

//! \brief Function for the computation of the values \f$ \Delta_j \f$
//! \param[in] t Vector of nodes \f$ t_j, j = 0,\dots,n \f$
//! \param[in] y Vector of values \f$ y_j, j = 0,\dots,n \f$
//! \return Vector \f$ (\Delta_1, \dots, \Delta_n) \f$ relative to (t, y) data
Vector delta(const Vector & t, const Vector & y) {
    assert(t.size() == y.size());
    assert(t.size() > 1);
    const size_t n = y.size() - 1;

    return (y.tail(n) - y.head(n)).cwiseQuotient(t.tail(n) - t.head(n));
}

//! \brief Class representing a pecevise cubic Hermite interpolant \f$ s \f$ with minmod-reconstructed slopes
class mmPCHI {
  public:
    //! \brief Construct the slopes \f$ c_j \f$ of the interpolant, from the data \f$ (t_j, y_j) \f$.
    //! \param[in] t Vector of nodes \f$ t_j, j = 0,\dots,n \f$
    //! \param[in] y Vector of values \f$ y_j, j = 0,\dots,n \f$
    mmPCHI(const Vector &t, const Vector &y);

    //! \brief Evaluate interpolant defined by (*this) at the points defined by x and store the result in v
    //! \param[in] x Vector of points \f$ x_i, i = 0,\dots,m \f$
    //! \return v Vector of values \f$ s(x_i), i = 0,\dots,m \f$
    Vector eval(const Vector &x) const;
  private:
    const Vector t, y;//!< Vectors containing nodes and values
    Vector c; //!< Vector containing slopes for the interpolant
};

mmPCHI::mmPCHI(const Vector &t, const Vector &y) : t(t), y(y), c(t.size()) {
    assert(t.size() == y.size());
    assert(t.size() > 1);

    const size_t n = t.size() - 1;

    Vector d = delta(t, y);

    c = Vector::Zero(n+1);
    c(0) = d(0);
    c(n) = d(n-1);

    for (size_t i = 1; i < n; ++i) {
        c(i) = minmod(d(i-1), d(i));
    }
}

Vector mmPCHI::eval(const Vector & x) const {
    Vector v(x.size());
    
    Vector xcopy = x;
    std::sort(xcopy.data(), xcopy.data()+xcopy.size());
    
    unsigned int i = 0;
    for (unsigned int j = 0; j < t.size()-1; ++j) {
        double t1 = t(j);
        double t2 = t(j+1);
        double y1 = y(j);

        double h = t2 - t1;
        double a1 = y(j+1) - y1;
        double a2 = a1 - h*c(j);
        double a3 = h*c(j+1) - a1 - a2;

        while( i < xcopy.size() && xcopy(i) <= t2 ) {
            double tx = (xcopy(i) - t1) / h;
            v(i) = y1 + (a1+(a2+a3*tx)*(tx-1.))*tx;
            i++;
        }
    }

    return v;
}


int main(int, char**) {
    //// PROBLEM 3 TEST

    auto f = [] (double t) -> double { return 1. / (1. + t*t); };
    // auto f = [] (double t) -> double { return 1. / (1. + std::abs(t)); };

    // Interval
    const double a = -5., b = 5;
    // Number of sampling nodes
    const unsigned int M = 10000;

    // Sampling nodes and values
    Vector x = Vector::LinSpaced(M,a,b);
    Vector v_ex(M);
    // Fill in exact values of f
    for (unsigned int i = 0; i<M; ++i) {
        v_ex(i) = f(x(i));
    }
    // Output table with error w.r.t number of nodes
    std::cout << "n" << "\t" << "L^inf err." << std::endl;
    for (unsigned int n = 4; n <= 4*2048; n=n<<1) {
        Vector t = Vector::LinSpaced(n,a,b);
        Vector y(n);
        for(unsigned int i = 0; i<n; ++i) {
            y(i) = f(t(i));
        }

        mmPCHI s(t,y);
        const Vector v = s.eval(x);
        std::cout << n << "\t" << (v - v_ex).lpNorm<Eigen::Infinity>()
                  << std::endl;
    }

    return 0;
}

