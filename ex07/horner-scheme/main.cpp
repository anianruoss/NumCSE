#include <complex>
#include <iomanip>
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "timer.h"


using namespace Eigen;

/*
 * @brief Evaluate a polynomial and its derivative using Horner scheme
 * @param[in] vector c of size $n$, coefficients of the polynomial p
 * @param[in] double x, where the polynomial has to be evaluated
 * @param[out] vector containing p(x),p'(x)
 */
template <typename CoeffVec>
Vector2d dpEvalHorner (const CoeffVec& c, const double x) {
    const unsigned int s = c.size();

    Vector2d val;
    val(0) = c[s-1];
    val(1) = c[s-1] * (s-1);

    for (unsigned int i = 1; i < s; ++i) {
        val(0) = x * val(0) + c[s-(i+1)];
    }

    for (unsigned int i = 1; i < (s-1); ++i) {
        val(1) = x * val(1) + (s-(i+1)) * c[s-(i+1)];
    }

    return val;
}

/*
 * @brief Evaluate a polynomial and its derivative using a naive implementation
 * @param[in] vector c of size $n$, coefficients of the polynomial p
 * @param[in] double x, where the polynomial has to be evaluated
 * @param[out] vector containing p(x),p'(x)
 */
template <typename CoeffVec>
Vector2d dpEvalNaive(const CoeffVec& c, const double x) {

    Vector2d val;
    //TODO

    return val;
}


int main() {
    std::vector<double> c = {3, 1, 5, 7, 9};
    double x = .123;

    // Check implementations
    Vector2d p, p_naive;
    p = dpEvalHorner(c,x);
    std::cout << "Using horner scheme:" << std::endl
              << "p(x) = " << p(0)
              << ", dp(x) = " << p(1) << std::endl;

    p_naive = dpEvalNaive(c,x);
    std::cout << "Using monomial approach:" << std::endl
              << "p(x) = " << p_naive(0)
              << ", dp(x) = " << p_naive(1) << std::endl;

    // Compare runtimes
    // TODO

    return 0;
}

