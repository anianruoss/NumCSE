#include "timer.h"
#include <iostream>
#include <iomanip>
#include <eigen3/Eigen/Dense>
#include <mgl2/mgl.h>

using namespace Eigen;

/* @brief Build an "arrow matrix" and compute A*A*y
 * Given vectors $a$ and $d$, returns A*A*x in $y$, where A is built from a, d
 * @param[in] d An n-dimensional vector
 * @param[in] a An n-dimensional vector
 * @param[in] x An n-dimensional vector
 * @param[out] y The vector y = A*A*x
 */
void arrow_matrix_2_times_x(const VectorXd &d,
                            const VectorXd &a,
                            const VectorXd &x,
                            VectorXd &y) {
    assert(d.size() == a.size() && a.size() == x.size() &&
           "Vector size must be the same!");
    int n = d.size();

    VectorXd d_head = d.head(n-1);
    VectorXd a_head = a.head(n-1);
    MatrixXd d_diag = d_head.asDiagonal();

    MatrixXd A(n,n);

    A << d_diag, a_head, a_head.transpose(), d(n-1);

    y = A*A*x;
}

/* @brief Build an "arrow matrix"
 * Given vectors $a$ and $b$, returns A*A*x in $y$, where A is build from a,d
 * @param[in] d An n-dimensional vector
 * @param[in] a An n-dimensional vector
 * @param[in] x An n-dimensional vector
 * @param[out] y The vector y = A*A*x
 */
void efficient_arrow_matrix_2_times_x(const VectorXd &d,
                                      const VectorXd &a,
                                      const VectorXd &x,
                                      VectorXd &y) {
    assert(d.size() == a.size() && a.size() == x.size() &&
           "Vector size must be the same!");
    int n = d.size();

    auto A_times_x = [&a, &d, n] (const VectorXd & x) {
        VectorXd Ax = (d.array() * x.array()).matrix();
        Ax.head(n-1) +=  x(n-1) * a.head(n-1);
        Ax(n-1) += a.head(n - 1).dot(x.head(n-1));

        return Ax;
    };

    y = A_times_x(A_times_x(x));
}

/* \brief Compute the runtime of arrow matrix multiplication.
 * Repeat tests 10 times, and output the minimal runtime
 * amongst all times. Test both the inefficient and the efficient
 * versions.
*/
void runtime_arrow_matrix() {
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
		   	VectorXd a = VectorXd::Random(n[i]);
            VectorXd d = VectorXd::Random(n[i]);
            VectorXd x = VectorXd::Random(n[i]);
            VectorXd y;

            timer.start();
            arrow_matrix_2_times_x(a,d,x,y);
            timer.stop();
			
            timer_eff.start();
            efficient_arrow_matrix_2_times_x(a,d,x,y);
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
        ref1[i] = 1e-8*pow(n[i],3);
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
    gr->Title("Runtime vs Matrix size");
    gr->SetRanges(n[0],n[0]*pow(2,nLevels-1),1e-6,1e+1);  gr->SetFunc("lg(x)","lg(y)");
    gr->Axis();
    gr->Plot(matSize,data1,"k +"); gr->AddLegend("original","k +");
    gr->Plot(matSize,data2,"r +"); gr->AddLegend("efficient","r +");
    gr->Plot(matSize,dataRef1,"k"); gr->AddLegend("O(n^3)","k");
    gr->Plot(matSize,dataRef2,"r"); gr->AddLegend("O(n)","r");
    gr->Label('x',"Matrix size [n]",0);
    gr->Label('y', "Runtime [s]",0);
    gr->Legend(2);
	gr->WriteFrame("bin/arrowmatvec_comparison.eps");
}


int main(void) {
    // Test vectors
    VectorXd a(5);
    a << 1., 2., 3., 4., 5.;
    VectorXd d(5);
    d <<1., 3., 4., 5., 6.;
    VectorXd x(5);
    x << -5., 4., 6., -8., 5.;
    VectorXd yi;

    std::cout << std::scientific << std::setprecision(3) << std::setw(15);

    // Run both functions
    arrow_matrix_2_times_x(a,d,x,yi);
    VectorXd ye(yi.size());
    efficient_arrow_matrix_2_times_x(a,d,x,ye);

    // Compute error
    double err = (yi - ye).norm();

    // Output error
    std::cout << "--> Correctness test." << std::endl;
    std::cout << "Error: " << err << std::endl;

    // Print out runtime
    std::cout << "--> Runtime test." << std::endl;
    runtime_arrow_matrix();

    // Final test: exit with error if error is too big
    double eps = std::numeric_limits<double>::denorm_min();
    exit(err < eps);

    return 0;
}

