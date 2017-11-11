#include <mgl2/mgl.h>
#include <Eigen/Dense>
#include "setFocus.hpp" // Contains set_focus
#include "utilities.hpp" // Contains important processing functions 
#include "fftLocal.hpp" // Contains FFT utilities

using namespace Eigen;

// TODO: implement the quantitative definition of high frequency content
// Computes high frequency content in matrix M
double high_frequency_content(const MatrixXd & M) {
    
    int n = M.rows(),m = M.cols();
    double V = 0;
    //TODO
    
    return V;
}


// plot the variation of high frequency content with focus parameter
void plotV(unsigned int N) {

    VectorXd x(N), y(N);

    for(unsigned int i = 0; i < N; ++i) {
        double V = high_frequency_content(
                    // Find 2D spectrum of matrix $\mathbf{B}(t)$
                    fft2r(
                        // Evaluate set\_focus at equidistant points
                        set_focus(5. / (N-1) * i)
                        )
                    .cwiseAbs()
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
		gr.WriteFrame("frequencyContent.eps");

}


int main() {

    //plotV(20);

}

// END OF FILE
