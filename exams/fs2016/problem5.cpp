#include <iostream>
#include <vector>
#include <Eigen/Dense>

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

//! \brief Evolves the vector y0 using the evolution operator \tilde{\Psi} with step-size y0
//! \tparam Operator type for evolution operator (e.g. lambda function type)
//! \param[in] Psi original evolution operator, must have operator(double, const Vector&)
//! \param[in] p parameter p for construction of Psi tilde
//! \param[in] h step-size
//! \param[in] y0 previous step
//! \return Evolved step \f$ \tilde{\Psi}^h y0 \f$
template <class Operator>
Vector psitilde(const Operator& Psi, unsigned int p, double h,
                const Vector & y0) {

    Vector out(y0);

    // TODO: problem 5b: implement psi tilde operator applied to h and y0

    return out;
}

//! \brief Evolves the vector y0 using the evolution operator from time 0 to T using equidistant steps
//! \tparam Operator type for evolution operator (e.g. lambda function type)
//! \param[in] Psi original evolution operator, must have operator(double, const Vector&)
//! \param[in] T final time
//! \param[in] y0 initial data
//! \param[in] N number of steps
//! \return Vector of all steps y_0, y_1, ...
template <class Operator>
std::vector<Vector> odeintequi(const Operator& Psi, double T, const Vector &y0,
                               int N) {

    std::vector<Vector> Y;

    // TODO: problem 5c: implement equidistant solver for evolution operator Psi

    return Y;
}

//! \brief Evolves the vector y0 using the evolution operator from time 0 to T using adaptive error control
//! \tparam Operator type for evolution operator (e.g. lambda function type)
//! \param[in] Psi original evolution operator, must have operator(double, const Vector&)
//! \param[in] T final time
//! \param[in] y0 initial data
//! \param[in] h0 initial step size
//! \param[in] p parameter p for construction of Psi tilde
//! \param[in] reltol relative tolerance for error control
//! \param[in] abstol absolute tolerance for error control
//! \param[in] p parameter p for construction of Psi tilde
//! \param[in] hmin minimal step size
//! \return Vector of all steps y_0, y_1, ...
template <class Operator>
std::vector<Vector> odeintssctrl(const Operator& Psi, double T,
                                 const Vector &y0, double h0, unsigned int p,
                                 double reltol, double abstol, double hmin) {
    double t = 0.;
    double h = h0;
    std::vector<Vector> Y;
    Y.push_back(y0);
    Vector y = y0;

    while( t < T && h > hmin ) {
        Vector y_high = Vector::Zero(y0.size()); // TODO: problem 5e: fix this line (implement high order discrete evolution)
        Vector y_low = Vector::Zero(y0.size()); // TODO: problem 5e: fix this line (implement low order discrete evolution)

        // Estimate error
        double err_est = 0.; // TODO: problem 5e: compute error estimator
        // Compute tolerance (min of reltol and abstol)
        double tol = 0.; // TODO: problem 5e: compute tolerance

        // Accept step if true (error smaller than tol)
        if( err_est < tol ) {
            t += h;

            y = y_high;
            Y.push_back(y_high);

        }
        // Predict step-size
        h *= 0.; // TODO: problem 5e: predict time step
        // Force time if needed
        h = std::min(T-t, h);
    }

    if( t < T ) std::cerr << "Failed to reach final time!" << std::endl;

    return Y;
}


int main(int, char**) {

    auto f = [] (const Vector &y) -> Vector { return Vector::Ones(1) + y*y; };
    Vector y0 = Vector::Zero(1);
    double T = 1.;
    auto y_ex = [] (double t) -> Vector { Vector y(1); y << tan(t); return y; };

    unsigned p = 1;

    //// PROBLEM 5d
    std::cout << "*** PROBLEM 5d:" << std::endl;

    // TODO: problem 5d: determine order of convergence of PsiTilde using odeintequi

    //// PROBLEM 5e
    std::cout << "*** PROBLEM 5e:" << std::endl;

    double h0 = 1./100.;
    std::vector<Vector> Y;
    double err = 0;

    // TODO: problem 5e: uncomment following lines to test odeintssctrl
//     Y = odeintssctrl(Psi, T, y0, h0, p, 10e-8, 10e-8, 10e-6);
//     err = (Y.back() - y_ex(1)).norm();
    std::cout << "Adaptive error control results:" << std::endl;
    std::cout << "Error" << "\t\t" << "No. Steps" << "\t" << "y(1)" << "\t"
              << "y(ex)" << std::endl;
//     std::cout << err << "\t" << Y.size() << "\t" << Y.back() << "\t" << y_ex(1) << std::endl;

    return 0;
}

