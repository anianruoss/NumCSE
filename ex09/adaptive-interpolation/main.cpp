#include "newtonIpol.hpp"
#include <mgl2/mgl.h>
#include <iostream>


using namespace Eigen;

template <class Function>
void adapPolyIpol(const Function &f, const double a, const double b,
                  const double tol, const unsigned N,
                  VectorXd &adaptive_nodes, VectorXd &error_vs_step_no) {
    // Generate sampling points and evaluate $f$ there
    VectorXd sampling_points = VectorXd::LinSpaced(N, a, b);
    VectorXd fvals_at_sampling_points = sampling_points.unaryExpr(f);

    adaptive_nodes(0) = 0.5*(b+a);
    VectorXd y = adaptive_nodes.unaryExpr(f);
    VectorXd poly;
    intPolyEval(adaptive_nodes, y, sampling_points, poly);

    while ((fvals_at_sampling_points - poly).maxCoeff()
            > tol * fvals_at_sampling_points.maxCoeff()) {
        y = adaptive_nodes.unaryExpr(f);
        intPolyEval(adaptive_nodes, y, sampling_points, poly);

        // calculate new node
        VectorXd::Index maxDiffIdx;
        double error = (fvals_at_sampling_points - poly).maxCoeff(&maxDiffIdx);
        adaptive_nodes.conservativeResize(adaptive_nodes.size()+1);
        adaptive_nodes(adaptive_nodes.size()-1) = sampling_points(maxDiffIdx);

        error_vs_step_no.conservativeResize(error_vs_step_no.size()+1);
        error_vs_step_no(error_vs_step_no.size()-1) = std::abs(error);
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
    gr.WriteFrame("plots/adapPolyIopl.eps");
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

    VectorXd adaptive_nodes = VectorXd::Zero(1);
    VectorXd error_vs_step_no(adaptive_nodes);

    adapPolyIpol(f1, a, b, tol, N, adaptive_nodes, error_vs_step_no);

    mglData daty;
    daty.Link(error_vs_step_no.data(), error_vs_step_no.size());

    mglGraph gr;
    gr.SetRanges(0, error_vs_step_no.size(), 0, error_vs_step_no.maxCoeff());
    gr.Axis();
    gr.Plot(daty, "b");
    gr.Label('x',"Step [N]", 0);
    gr.Label('y', "Max. Error", 0);
    gr.Title("Maximum Interpolation Error");
    gr.WriteFrame("plots/adapPolyIpol-maxError.eps");

    return 0;
}

