#include "newtonIpol.hpp"
#include <mgl2/mgl.h>
#include <iostream>
#include <string>


using namespace Eigen;

template <class Function>
void adapPolyIpol(const Function &f, const double a, const double b,
                  const double tol, const unsigned N,
                  VectorXd &adaptive_nodes, VectorXd &error_vs_step_no,
                  std::string frameName) {
    // Generate sampling points and evaluate $f$ there
    VectorXd sampling_points = VectorXd::LinSpaced(N, a, b);
    VectorXd fvals_at_sampling_points = sampling_points.unaryExpr(f);

    adaptive_nodes(0) = 0.5*(b+a);
    VectorXd y, poly;

    double fMax = fvals_at_sampling_points.cwiseAbs().maxCoeff();

    for (unsigned int i = 0; i < N; ++i) {
        y = adaptive_nodes.unaryExpr(f);
        intPolyEval(adaptive_nodes, y, sampling_points, poly);

        // calculate maximum error
        VectorXd::Index maxErrorIdx;
        VectorXd errors = (fvals_at_sampling_points - poly).cwiseAbs();
        double maxError = errors.maxCoeff(&maxErrorIdx);

        // check termination criterium
        if (maxError <= tol * fMax) {
            break;
        }

        // add node to set and save the error
        adaptive_nodes.conservativeResize(adaptive_nodes.size()+1);
        adaptive_nodes(adaptive_nodes.size()-1) = sampling_points(maxErrorIdx);

        error_vs_step_no.conservativeResize(error_vs_step_no.size()+1);
        error_vs_step_no(error_vs_step_no.size()-1) = maxError;

        if (i == N-1) {
            std::cerr << "Desired accuracy could not be reached." << std::endl;
        }
    }

    // sort nodes for plotting
    std::sort(adaptive_nodes.data(),
              adaptive_nodes.data() + adaptive_nodes.size(),
    [] (double a, double b) {
        return a < b;
    }
             );
    y = adaptive_nodes.unaryExpr(f);

    mglData refx, refy;
    refx.Link(sampling_points.data(), sampling_points.size());
    refy.Link(fvals_at_sampling_points.data(), sampling_points.size());

    mglData datx, daty;
    datx.Link(adaptive_nodes.data(), adaptive_nodes.size());
    daty.Link(y.data(), adaptive_nodes.size());

    mglGraph gr;
    gr.SetRanges(0, 1, -1, 1);
    gr.Axis();
    gr.Plot(refx, refy, "g *");
    gr.AddLegend("Original Function", "g *");
    gr.Plot(datx, daty, "b");
    gr.AddLegend("Interpolation", "b");
    gr.Legend(0);
    gr.Title("Adaptive Polynomial Interpolation");
    std::string name = "plots/adapPolyIpol-" + frameName + ".eps";
    gr.WriteFrame(name.c_str());
}


int main() {
    auto f1 = [] (double t) {
        return std::sin(std::exp(2.*t));
    };

    auto f2 = [] (double t) {
        return std::sqrt(t)/(1 + 16.*t*t);
    };

    double a = 0;
    double b = 1;
    double tol = 1e-6;
    unsigned int N = 1000;

    VectorXd adaptive_nodes_f1 = VectorXd::Zero(1);
    VectorXd adaptive_nodes_f2 = VectorXd::Zero(1);
    VectorXd error_vs_step_no_f1, error_vs_step_no_f2;

    adapPolyIpol(f1, a, b, tol, N, adaptive_nodes_f1, error_vs_step_no_f1, "f1");
    adapPolyIpol(f2, a, b, tol, N, adaptive_nodes_f2, error_vs_step_no_f2, "f2");

    VectorXd ref1 = VectorXd::LinSpaced(error_vs_step_no_f1.size(), 1,
                                        error_vs_step_no_f1.size());
    VectorXd ref2 = VectorXd::LinSpaced(error_vs_step_no_f2.size(), 1,
                                        error_vs_step_no_f2.size());

    mglData dat1x, dat2x;
    dat1x.Link(ref1.data(), ref1.size());
    dat2x.Link(ref2.data(), ref2.size());

    mglData dat1y, dat2y;
    dat1y.Link(error_vs_step_no_f1.data(), error_vs_step_no_f1.size());
    dat2y.Link(error_vs_step_no_f2.data(), error_vs_step_no_f2.size());

    double xMax = std::max(ref1.size(), ref2.size());
    double yMax	= std::max(error_vs_step_no_f1.maxCoeff(),
                           error_vs_step_no_f2.maxCoeff());
    double yMin = std::min(error_vs_step_no_f1.minCoeff(),
                           error_vs_step_no_f2.minCoeff());

    mglGraph gr;
    gr.Title("Maximum Interpolation Error");
    gr.SetRanges(1, xMax, yMin, yMax);
    gr.SetFunc("x","lg(y)");
    gr.Axis();
    gr.Plot(dat1x, dat1y, "b+");
    gr.AddLegend("f_1(t) = sin(e^{2t})","b+");
    gr.Plot(dat2x, dat2y, "rs");
    gr.AddLegend("f_2(t) = \\sqrt{t} / (1 + 16t^2)","rs");
    gr.Label('x',"Step [N]", 0);
    gr.Label('y', "Max. Error", 0);
    gr.Legend();
    gr.WriteFrame("plots/adapPolyIpol-maxError.eps");

    return 0;
}

