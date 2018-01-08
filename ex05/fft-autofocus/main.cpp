#include "src/setFocus.hpp" // Contains set_focus
#include "src/utilities.hpp" // Contains important processing functions 
#include "src/fftLocal.hpp" // Contains FFT utilities

#include <mgl2/mgl.h>
#include <eigen3/Eigen/Dense>

using namespace Eigen;

// Computes high frequency content in matrix M
double high_frequency_content(const MatrixXd & M) {
    const int n = M.rows(), m = M.cols();
    double V{0};

    for (int k1 = 0; k1 < n; ++k1) {
        for (int k2 = 0; k2 < m; ++ k2) {
            double a = n/2.0 - std::abs(k1 - n/2.0);
            double b = m/2.0 - std::abs(k2 - m/2.0);
            V += (a*a + b*b) * M(k1,k2) * M(k1,k2);
        }
    }

    return V;
}

// Plots the variation of high frequency content with focus parameter
void plotV(unsigned int N) {

    VectorXd x(N), y(N);

    for (unsigned int i = 0; i < N; ++i) {
        double V = high_frequency_content(
                       // Find 2D spectrum of matrix $\mathbf{B}(t)$
                       fft2r(
                           // Evaluate set\_focus at equidistant points
                           set_focus(5. / (N-1) * i)
                       ).cwiseAbs()
                   );
        x(i) = 5. / (N-1) * i;
        y(i) = V;
        std::cout << x(i) << "\t" << y(i) << std::endl;
    }

    mglData datx, daty;
    datx.Link(x.data(), N);
    daty.Link(y.data(), N);
    mglGraph gr;
    gr.Title("High frequency content");
    gr.SetRanges(0,5,0,4e+16);
    gr.Axis();
    gr.Plot(datx, daty, "r+");
    gr.Label('x',"$f$",0);
    gr.Label('y',"$V(\\mathbf{B}(f))$",0);
    gr.WriteFrame("plots/frequencyContent.eps");
}


int main() {
    for (int f = 0; f < 4; ++f) {
        plot_freq(f);
        save_image(f);
    }

    plotV(20);

    return 0;
}

