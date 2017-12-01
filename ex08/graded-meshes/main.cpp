#include <eigen3/Eigen/Dense>
#include <mgl2/mgl.h>
#include <iostream>
#include <string>


using namespace Eigen;

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
    const int nInterpPts = T.size();
    VectorXd Y = VectorXd::Zero(nInterpPts);

    for (int i = 0; i < nInterpPts; ++i) {
        Y(i) = std::pow(T(i), a);
    }

    VectorXd evalInterp = evalPiecewiseInterp(T, Y, evalT);

    double maxError = 0;
    for (int i = 0; i < evalT.size(); ++i) {
        double error = std::abs(evalInterp(i) - std::pow(evalT(i), a));
        if (error > maxError) {
            maxError = error;
        }
    }

    return maxError;
}

void plotInterpolation(VectorXd T, VectorXd evalT, const char* plotName) {
    const char colours[] = {'r', 'b', 'm', 'y', 'q', 'h', 'p'};

    int nParamVals = 7;
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

        std::string title =	"alpha = " + std::to_string(paramVals(i));
        const char c = *(colours+i);

        gr.Plot(datx, daty, &c);
        gr.AddLegend(title.c_str(), &c);
        gr.Plot(refx, refy, "g *");
    }

    gr.AddLegend("Reference Points", "g *");
    gr.Title("Linear Interpolation of t^{alpha}");
    gr.Legend(1);
    gr.WriteFrame(plotName);
}


int main() {
    {
        int nInterpNodes = 8;
        int nEvalPts = (1 << 9);
        VectorXd T = VectorXd::LinSpaced(nInterpNodes, 0, 1);
        VectorXd evalT = VectorXd::LinSpaced(nEvalPts, 0, 1);

        plotInterpolation(T, evalT, "uniformMesh-linearInterpolation.eps");

        int nParamVals = 100;
        VectorXd paramVals = VectorXd::LinSpaced(nParamVals, 0, 2);
        VectorXd maxErrors = VectorXd::Zero(nParamVals);

        for (int i = 0; i < nParamVals; ++i) {
            maxErrors(i) = maxInterpError(paramVals(i), T, evalT);
        }

        mglData datx, daty;
        datx.Link(paramVals.data(), nParamVals);
        daty.Link(maxErrors.data(), nParamVals);
        mglGraph gr;
        gr.SetRanges(0, 2, 0, 1);
        gr.Axis();
        gr.Plot(datx, daty);
        gr.WriteFrame("uniformMesh-maxInterpolationError-varAlpha.eps");
    }

    return 0;
}

