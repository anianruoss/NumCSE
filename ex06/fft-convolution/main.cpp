#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <unsupported/Eigen/FFT>
#include <mgl2/mgl.h>

#include <typeinfo>

using namespace Eigen;



VectorXd discreteConvolution(const VectorXd &x, const VectorXd &y) {
    using index_t = MatrixXd::Index;
    index_t m = x.size(), n = y.size();

    VectorXd out(m+n-1);

    for (index_t k = 0; k < m+n-1; ++k) {
        for (index_t j = 0; j < m; ++j) {
            if ((k-j >= 0) && (k-j < n)) {
                out(k) += x(j) * y(k-j);
            }
        }
    }

    return out;
}

VectorXd PointB(const VectorXd &x, const VectorXd &y) {
    VectorXd out;
    // TODO

    return out;
}

VectorXd PointC(const VectorXd &x, const VectorXd &y) {
    VectorXd out;
    // TODO

    return out;
}


int main() {
    srand(time(NULL));
    VectorXd input = VectorXd::Zero(100);
    VectorXd noise = VectorXd::Random(100);
    input += (noise.array() - 0.5).matrix() / 5;

    // define truncated gaussian filter
    int N = 5;
    double a = .5;
    VectorXd gaussFilter(2*N+1);
    for (int j=0; j < 2*N+1; j++) {
        gaussFilter(j) = std::sqrt(a/3.1416) * std::exp(-a * ((j-N)) * (j-N));
    }

    // Plotting
    mglGraph gr, grA, grB, grC;
    mglData dat, datA, datB, datC;
    dat.Link(input.data(), input.size());
    gr.Plot(dat);
    gr.WriteFrame("bin/data/noisySignal.png");
    auto convA = discreteConvolution(input, gaussFilter);
    datA.Link(convA.data(), convA.size());
    gr.Plot(datA);
    grA.Plot(datA);
    grA.WriteFrame("bin/data/filteredA.png");
    auto convB = PointB(input, gaussFilter);
    datB.Link(convB.data(), convB.size());
    gr.Plot(datB);
    grB.Plot(datB);
    grB.WriteFrame("bin/data/filteredB.png");
    auto convC = PointC(input, gaussFilter);
    datC.Link(convC.data(), convC.size());
    gr.Plot(datC);
    grC.Plot(datC);
    grC.WriteFrame("bin/data/filteredC.png");
    gr.WriteFrame("bin/data/noisy+filters.png");

    return 0;
}

