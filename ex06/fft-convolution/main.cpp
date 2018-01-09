#include <eigen3/Eigen/Dense>
#include <mgl2/mgl.h>
#include <unsupported/Eigen/FFT>

#include <cmath>
#include <iostream>

using namespace Eigen;
using index_t = MatrixXd::Index;

VectorXd discreteConv(const VectorXd &x, const VectorXd &y) {
    const int m = x.size(), n = y.size();
    const int L = m + n - 1;
    VectorXd out = VectorXd::Zero(L);

    for (int i = 0; i < L; ++i) {
        for (int k = std::max(0,i+1-m); k < std::min(i+1,n); ++k) {
            out(i) += x(i-k)*y(k);
        }
    }

    return out;
}

VectorXd overlappingConv(const VectorXd &x, const VectorXd &y) {
    const index_t m = x.size(), n = y.size();
    VectorXd ym = VectorXd::Zero(m);
    ym.head(n) = y;

    FFT<double> fft;
    VectorXcd fft_x = fft.fwd(x);
    VectorXcd fft_ym = fft.fwd(ym);
    VectorXcd tmp = fft_x.cwiseProduct(fft_ym);

    return fft.inv(tmp);
}

VectorXd discreteFFTConv(const VectorXd &x, const VectorXd &y) {
    const index_t m = x.size(), n = y.size();
    const index_t L = m + n - 1;

    VectorXd xL = VectorXd::Zero(L);
    xL.head(m) = x;
    VectorXd yL = VectorXd::Zero(L);
    yL.head(n) = y;

    FFT<double> fft;
    VectorXcd fft_xL = fft.fwd(xL);
    VectorXcd fft_yL = fft.fwd(yL);
    VectorXcd tmp = fft_xL.cwiseProduct(fft_yL);

    return fft.inv(tmp);
}

int main() {
    srand(time(NULL));
    VectorXd input = VectorXd::Zero(100);
    VectorXd noise = VectorXd::Random(100);
    input += (noise.array() - 0.5).matrix() / 5;

    // define truncated gaussian filter
    const int N = 5;
    const double a = .5;
    VectorXd gaussFilter(2*N+1);
    for (int j = 0; j < 2*N+1; ++j) {
        gaussFilter(j) = std::sqrt(a/3.1416) * std::exp(-a * ((j-N)) * (j-N));
    }

    // Plotting
    mglGraph gr, grA, grB, grC;
    mglData dat, datA, datB, datC;
    dat.Link(input.data(), input.size());
    gr.Plot(dat);
    gr.WriteFrame("plots/noisySignal.png");
    VectorXd convA = discreteConv(input, gaussFilter);
    datA.Link(convA.data(), convA.size());
    gr.Plot(datA);
    grA.Plot(datA);
    grA.WriteFrame("plots/filteredA.png");
    VectorXd convB = overlappingConv(input, gaussFilter);
    datB.Link(convB.data(), convB.size());
    gr.Plot(datB);
    grB.Plot(datB);
    grB.WriteFrame("plots/filteredB.png");
    VectorXd convC = discreteFFTConv(input, gaussFilter);
    datC.Link(convC.data(), convC.size());
    gr.Plot(datC);
    grC.Plot(datC);
    grC.WriteFrame("plots/filteredC.png");
    gr.WriteFrame("plots/noisy+filters.png");

    return 0;
}

