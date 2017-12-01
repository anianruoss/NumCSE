#include <eigen3/Eigen/Dense>
#include <mgl2/mgl.h>
#include <iostream>
#include <string>


using namespace Eigen;

// returns the values of the piecewise linear interpolant in evalT.
VectorXd evalPiecewiseInterp(const VectorXd &T, const VectorXd &Y,
                             const VectorXd &evalT) {
    const int n = evalT.size();
    VectorXd out = VectorXd::Zero(n);

    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < T.size()-1; ++i) {
            if (evalT(k) <= T(i+1)) {
                double slope = (Y(i+1) - Y(i)) / (T(i+1) - T(i));
                out(k) = Y(i) + slope * (evalT(k) - T(i));
                break;
            }
        }
    }

    return out;
}

double maxInterpError(double a, VectorXd T, VectorXd evalT) {
    double maxError = 0;

    // TODO: Implement me

    return maxError;
}

void plotInterpolation(VectorXd T, VectorXd evalT) {
    const char colours[] = {'k', 'r', 'b', 'm', 'y', 'q'};

    int nParamVals = 4;
    VectorXd paramVals = VectorXd::LinSpaced(nParamVals, 0, 2);

    const int nEvalPts = evalT.size();
    const int nInterpPts = T.size();
    VectorXd Y = VectorXd::Zero(nInterpPts);

    mglGraph gr;
    gr.SetRanges(0,1,0,1);
    gr.Axis();

    for (int i = 0; i < nParamVals; ++i) {
        for (int k = 0; k < nInterpPts; ++k) {
            Y(k) = std::pow(T(k), paramVals(i));
        }

        VectorXd evalInterp = evalPiecewiseInterp(T, Y, evalT);

        mglData refx, refy;
        refx.Link(T.data(), nInterpPts);
        refy.Link(Y.data(), nInterpPts);

        mglData datx, daty;
        datx.Link(evalT.data(), nEvalPts);
        daty.Link(evalInterp.data(), nEvalPts);

        std::string title =	"alpha: " + std::to_string(paramVals(i));
        const char c = *(colours+i);

        gr.Plot(datx, daty, &c);
        gr.AddLegend(title.c_str(), &c);
        gr.Plot(refx, refy, "g *");
    }

    gr.AddLegend("Reference Points", "g *");
    gr.Legend(1);
    gr.WriteFrame("linearInterpolation.eps");
}


int main() {

    int nInterpNodes = 8;
    int nEvalPts = (1 << 9);
    VectorXd T = VectorXd::LinSpaced(nInterpNodes, 0, 1);
    VectorXd evalT = VectorXd::LinSpaced(nEvalPts, 0, 1);

    int nParamVals = 100;
    VectorXd paramVals = VectorXd::LinSpaced(nEvalPts, 0, 2);

    plotInterpolation(T, evalT);

    return 0;
}

