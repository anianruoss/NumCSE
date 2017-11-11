#include <Eigen/Dense>
#include "setFocus.hpp" // Contains set_focus
#include "pgm.hpp" // Contains PGMObject
#include "fftLocal.hpp" // Contains FFT utilities

using namespace Eigen;

void save_image(double focus) {
    // Create empty object
    PGMObject q;

    // Set data using function "set\_data"
    // Data obtained from "set\_focus"
    MatrixXd B = set_focus(focus);
    q.set_data(B);

    // Create and save file
    std::stringstream ss;
    ss << "image_focus="
       << (int) focus
       << ".pgm";
    std::ofstream file(ss.str());
    file << q;
}

void plot_freq(double focus) {
    int a = 0;
    int b = 8000;
    auto clamp = [a,b] (double x) {
        return x < a ? a : x > b ? b : x;
    };

    MatrixXd D = fft2r(set_focus(focus))
            .cwiseAbs()
            .unaryExpr(clamp) / b;

    // Plot values of $\mathbf{X}$.
    mglData Xd(D.cols(), D.rows(), D.data());
    mglGraph gr;
    gr.Colorbar("bcwyr");
    std::stringstream ss;
    ss << "Spectrum with f = "
       << focus
       << ".";
    gr.Title(ss.str().c_str());
    gr.Axis(); gr.Tile(Xd, "bcwyr");
    std::stringstream ss2;
    ss2 << "spectrum_focus="
        << focus
        << ".png";
    gr.WritePNG(ss2.str().c_str());
}
