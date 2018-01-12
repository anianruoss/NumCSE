#include "timer.h"
#include <eigen3/Eigen/Dense>
#include <mgl2/mgl.h>

#include <complex>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace Eigen;

/*
 * @brief Evaluate a polynomial and its derivative using Horner scheme
 * @param[in] vector c of size $n$, coefficients of the polynomial p
 * @param[in] double x, where the polynomial has to be evaluated
 * @param[out] vector containing p(x),p'(x)
 */
template <typename CoeffVec>
Vector2d dpEvalHorner(const CoeffVec& c, const double x) {
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
    const unsigned int s = c.size();
    Vector2d val = Vector2d::Zero();

    for (unsigned int i = 0; i < s; ++i) {
        val(0) += std::pow(x, i)*c[i];
    }

    for (unsigned int i = 1; i < s; ++i) {
        val(1) += std::pow(x,i-1)*c[i]*i;
    }

    return val;
}


int main() {
    std::vector<double> c {3, 1, 5, 7, 9};
    const double x = .123;

    // Check implementations
    Vector2d p, p_naive;
    p = dpEvalHorner(c,x);
    std::cout << "Using horner scheme:" << std::endl
              << "p(x) = " << p(0)
              << ", dp(x) = " << p(1)
              << std::endl << std::endl;

    p_naive = dpEvalNaive(c,x);
    std::cout << "Using monomial approach:" << std::endl
              << "p(x) = " << p_naive(0)
              << ", dp(x) = " << p_naive(1)
              << std::endl << std::endl;

    // Compare runtimes
    const unsigned repeats = 10;
    double timesNaive[19], timesHorner[19], error[19];
    const size_t sep = 20;

    std::cout << std::setw(10) << "n"
              << std::setw(sep) << "horner scheme"
              << std::setw(sep) << "monomial approach"
              << std::setw(sep) << "error"
              << std::fixed << std::setprecision(10)
              << std::endl;

    for (unsigned k = 2; k <= 20; ++k) {
        const unsigned n = std::pow(2,k);
        VectorXd cVec = VectorXd::Random(n);
        std::vector<double> cTest(cVec.data(), cVec.data() + cVec.size());

        Timer tm_slow, tm_fast;

        for (unsigned r = 0; r < repeats; ++r) {
            tm_slow.start();
            p_naive = dpEvalNaive(cTest,x);
            tm_slow.stop();

            tm_fast.start();
            p = dpEvalHorner(cTest,x);
            tm_fast.stop();
        }

        timesHorner[k-2] = tm_fast.min();
        timesNaive[k-2] = tm_slow.min();
        error[k-2] = (p - p_naive).norm();

        std::cout << std::setw(10) << n
                  << std::setw(sep) << timesHorner[k-2]
                  << std::setw(sep) << timesNaive[k-2]
                  << std::setw(sep) << error[k-2]
                  << std::endl;
    }

    // plot results with MathGL
    double ref[19];
    for (unsigned int i = 0; i < 19; ++i) {
        ref[i] = (1 << (i+2));
    }

    mglData dataRef;
    dataRef.Link(ref, 19);

    mglData data1, data2;
    data1.Link(timesHorner, 19);
    data2.Link(timesNaive, 19);

    mglGraph *gr = new mglGraph;
    gr->Title("Runtime vs Polynomial Order");
    gr->SetRanges(2,std::pow(2,20),1e-6,1);
    gr->SetFunc("lg(x)","lg(y)");
    gr->Axis();
    gr->Plot(dataRef,data1,"r+");
    gr->AddLegend("Horner","r+");
    gr->Plot(dataRef,data2,"k+");
    gr->AddLegend("Monomial", "k+");
    gr->Label('x',"Polynomial Order [n]",0);
    gr->Label('y', "Runtime [s]",0);
    gr->Legend(2);
    gr->WriteFrame("plots/horner-runtimes.eps");

    return 0;
}

