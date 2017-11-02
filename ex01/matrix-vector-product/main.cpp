#include "timer.h"
#include <iostream>
#include <iomanip>
#include <eigen3/Eigen/Dense>
#include <mgl2/mgl.h>

using namespace Eigen;


/* \brief compute $\mathbf{A}\mathbf{x}$
 * \mathbf{A} is defined by $(\mathbf{A})_{i,j} := \min {i,j}$
 * \param[in] x vector x for computation of A*x = y
 * \param[out] y = A*x
 */
void multAminSlow(const VectorXd &x, VectorXd &y) {
    unsigned int n = x.size();

    VectorXd one = VectorXd::Ones(n);
    VectorXd linsp = VectorXd::LinSpaced(n,1,n);
    y = ((one*linsp.transpose()).cwiseMin(linsp*one.transpose())) * x;
}

/* \brief compute $\mathbf{A}\mathbf{x}$
 * This function has optimal complexity.
 * \mathbf{A} is defined by $(\mathbf{A})_{i,j} := \min {i,j}$
 * \param[in] x vector x for computation of A*x = y
 * \param[out] y = A*x
 */
void multAmin(const VectorXd &x, VectorXd &y) {
    unsigned int n = x.size();
    double sum = x.sum();

    y = VectorXd::Zero(n);
    y(0) = sum;

    for (unsigned int i = 1; i < n; ++i) {
        sum -= x(i-1);
        y(i) = y(i-1) + sum;
    }
}

void runtime_multAmin() {
    unsigned int nLevels = 10;
    unsigned int *n = new unsigned int[nLevels];
    double *minTime = new double[nLevels];
    double *minTimeEff = new double[nLevels];

    n[0] = 4;
    for (unsigned int i = 1; i < nLevels; ++i) {
        n[i] = 2*n[i-1];
    }

    std::cout << std::setw(8) << "n"
              << std::setw(15) << "original"
              << std::setw(15) << "efficient" << std::endl;

    unsigned int repeats = 10;

    for (unsigned int i = 0; i < nLevels; ++i) {
        Timer timer, timer_eff;

        for (unsigned int r = 0; r < repeats; ++r) {
            VectorXd x = VectorXd::Random(n[i]);
            VectorXd y;

            timer.start();
            multAminSlow(x,y);
            timer.stop();

            timer_eff.start();
            multAmin(x,y);
            timer_eff.stop();
        }

        minTime[i] = timer.min();
        minTimeEff[i] = timer_eff.min();

        std::cout << std::setw(8) << n[i]
                  << std::scientific << std::setprecision(3)
                  << std::setw(15) << minTime[i]
                  << std::setw(15) << minTimeEff[i] << std::endl;

    }

    // plot results with MathGL
    double nMgl[nLevels];
    double ref1[nLevels], ref2[nLevels];
    for (unsigned int i = 0; i < nLevels; ++i) {
        nMgl[i] = n[i];
        ref1[i] = 1e-8*pow(n[i],2);
        ref2[i] = 1e-8*n[i];
    }

    mglData matSize;
    matSize.Link(nMgl, nLevels);

    mglData data1, data2;
    mglData dataRef1, dataRef2;
    data1.Link(minTime, nLevels);
    data2.Link(minTimeEff, nLevels);
    dataRef1.Link(ref1,nLevels);
    dataRef2.Link(ref2,nLevels);

    mglGraph *gr = new mglGraph;
    gr->Title("Runtime vs Vector size");
    gr->SetRanges(n[0],n[0]*pow(2,nLevels-1),1e-8,1e-1);
    gr->SetFunc("lg(x)","lg(y)");
    gr->Axis();
    gr->Plot(matSize,data1,"k +");
    gr->AddLegend("original","k +");
    gr->Plot(matSize,data2,"r +");
    gr->AddLegend("efficient","r +");
    gr->Plot(matSize,dataRef1,"k");
    gr->AddLegend("O(n^2)","k");
    gr->Plot(matSize,dataRef2,"r");
    gr->AddLegend("O(n)","r");
    gr->Label('x',"Vector size [n]",0);
    gr->Label('y', "Runtime [s]",0);
    gr->Legend(2);
    gr->WriteFrame("bin/multAmin_comparison.eps");
}


int main(void) {
    // Testing correctness of the code
    unsigned int M = 10;
    VectorXd xa = VectorXd::Random(M);
    VectorXd ys, yf;

    multAmin(xa, yf);
    multAminSlow(xa, ys);

    std::cout << "--> Correctness test." << std::endl;
    std::cout << "||ys-yf|| = " << (ys - yf).norm() << std::endl;

    std::cout << std::endl << "--> Runtime test." << std::endl;
    runtime_multAmin();

    MatrixXd B = MatrixXd::Zero(M,M);
    for (unsigned int i = 0; i < M; ++i) {
        B(i,i) = 2;
        if (i < M-1) B(i+1,i) = -1;
        if (i > 0) B(i-1,i) = -1;
    }
    B(M-1,M-1) = 1;

    // restore default stream
    std::cout.copyfmt(std::ios(NULL));
    std::cout << std::endl << "B = " << std::endl << B << std::endl;

    std::cout << std::endl << "--> Test B = inv(A)" << std::endl;
    VectorXd x = VectorXd::Random(M), y;

    std::cout << std::scientific << std::setprecision(3);
    multAminSlow(B*x,y);
    std::cout << std::setw(10) << "Original: "
              << std::setw(10) << "|y-x| = "
              << std::setw(10) << (y-x).norm()
              << std::endl;

    multAmin(B*x,y);
    std::cout << std::setw(10) << "Efficient:"
              << std::setw(10) << "|y-x| = "
              << std::setw(10) << (y-x).norm()
              << std::endl;

    return 0;
}

