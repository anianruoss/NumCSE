// run with: g++ -std=gnu++11 -I /usr/include/eigen3/ -g conv.cpp -lmgl 
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <unsupported/Eigen/FFT>
#include <mgl2/mgl.h>

using namespace Eigen;

VectorXd PointA(const VectorXd &x, const VectorXd &y) {
	VectorXd out;
	// TODO
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
	gr.WriteFrame("noisySignal.png");
	auto convA = PointA(input, gaussFilter);
  	datA.Link(convA.data(), convA.size());
	gr.Plot(datA);
	grA.Plot(datA);
	grA.WriteFrame("filteredA.png");
	auto convB = PointB(input, gaussFilter);
  	datB.Link(convB.data(), convB.size());
	gr.Plot(datB);
	grB.Plot(datB);
	grB.WriteFrame("filteredB.png");
	auto convC = PointC(input, gaussFilter);
  	datC.Link(convC.data(), convC.size());
	gr.Plot(datC);
	grC.Plot(datC);
	grC.WriteFrame("filteredC.png");
	gr.WriteFrame("noisy+filters.png");
}
