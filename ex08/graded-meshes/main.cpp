#include <eigen3/Eigen/Dense>
#include <mgl2/mgl.h>
#include <iostream>
#include <string>


using namespace Eigen;

VectorXd evalPiecewiseInterp(const VectorXd &T, const VectorXd &Y,
                             const VectorXd &evalT) {
    const int n = evalT.size();
    const int m = T.size();

    VectorXd out = VectorXd::Zero(n);
    VectorXd slope = VectorXd::Zero(m-1);
    slope = (Y.tail(m-1)-Y.head(m-1)).cwiseQuotient(T.tail(m-1)-T.head(m-1));

    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < m-1; ++i) {
            if (T(i) <= evalT(k) && evalT(k) <= T(i+1)) {
                out(k) = Y(i) + slope(i) * (evalT(k) - T(i));
                break;
            }
        }
    }

    return out;
}

double maxInterpError(double a, VectorXd T, VectorXd evalT) {
    auto f = [a] (double t) {
        return std::pow(t,a);
    };

    VectorXd Y = T.unaryExpr(f);
    VectorXd evalInterp = evalPiecewiseInterp(T, Y, evalT);

    return (evalT.unaryExpr(f) - evalInterp).cwiseAbs().maxCoeff();
}

void plotInterpolation(VectorXd T, VectorXd evalT, const char* plotName,
                       bool graded = false) {
    const char colours[] = {'r', 'b', 'm', 'y', 'q', 'h', 'p'};

    int nParamVals = 7;
    VectorXd paramVals = VectorXd::LinSpaced(nParamVals, 0.123456, 1.987654);

    const int nEvalPts = evalT.size();
    const int nInterpPts = T.size();
    VectorXd Y = VectorXd::Zero(nInterpPts);

    mglGraph gr;
    gr.SetRanges(0,1,0,1);
    gr.Axis();

    for (int i = 0; i < nParamVals; ++i) {
        VectorXd tempT(T);

        if (graded) {
            for (int k = 0; k < nInterpPts; ++k) {
                tempT(k) = std::pow(T(k), 2./paramVals(i));
            }
        }

        for (int k = 0; k < nInterpPts; ++k) {
            Y(k) = std::pow(tempT(k), paramVals(i));
        }

        VectorXd evalInterp = evalPiecewiseInterp(tempT, Y, evalT);

        mglData refx, refy;
        refx.Link(tempT.data(), nInterpPts);
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
        int nEvalPts = 1 << 9;
        VectorXd T = VectorXd::LinSpaced(nInterpNodes, 0, 1);
        VectorXd evalT = VectorXd::LinSpaced(nEvalPts, 0, 1);

        plotInterpolation(T, evalT, "plots/uniformMesh-linInterpolation.eps");

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
        gr.Label('x', "alpha", 0);
        gr.Label('y', "max. error", 0);
        gr.Title("Maximum Interpolation Error");
        gr.WriteFrame("plots/uniformMesh-maxInterpolationError-varAlpha.eps");
    }

    {
        double alpha = 0.54321;
        int nTests = 10;
        int nEvalPts = 1 << 12;
        VectorXd evalT = VectorXd::LinSpaced(nEvalPts, 0, 1);
        VectorXd maxErrors = VectorXd::Zero(nTests);

        for (int i = 0; i < nTests; ++i) {
            int nInterpNodes = 1 << i;
            VectorXd T = VectorXd::LinSpaced(nInterpNodes, 0, 1);
            maxErrors(i) = std::log(maxInterpError(alpha, T, evalT));
        }

        mglData daty;
        daty.Link(maxErrors.tail(nTests).data(), nTests);
        mglGraph gr;
        gr.SetRanges(0, nTests, -6, 0);
        gr.Axis();
        gr.Plot(daty);
        gr.Label('x', "mesh size (2^{n})", 0);
        gr.Label('y', "log(max. error)", 0);
        gr.Title("Maximum Interpolation Error");
        gr.WriteFrame("plots/uniformMesh-logMaxInterpolationError-varN.eps");
    }

    {
        int nInterpNodes = 8;
        int nEvalPts = 1 << 9;
        VectorXd T = VectorXd::LinSpaced(nInterpNodes, 0, 1);
        VectorXd evalT = VectorXd::LinSpaced(nEvalPts, 0, 1);

        plotInterpolation(T, evalT, "plots/gradedMesh-linInterpolation.eps",
                          true);
    }

    {
        double a = 0.5;
        double b = 2./a;

        int nTests = 10;
        int nEvalPts = 1 << 12;
        VectorXd evalT = VectorXd::LinSpaced(nEvalPts, 0, 1);
        VectorXd maxErrors = VectorXd::Zero(nTests);

        for (int i = 0; i < nTests; ++i) {
            int nInterpNodes = 1 << i;
            VectorXd T = VectorXd::LinSpaced(nInterpNodes, 0, 1);

            for (int k = 0; k < nInterpNodes; ++k) {
                T(k) = std::pow(T(k), b);
            }

            maxErrors(i) = std::log(maxInterpError(a, T, evalT));
        }

        mglData daty;
        daty.Link(maxErrors.tail(nTests).data(), nTests);
        mglGraph gr;
        gr.SetRanges(0, nTests, -14, 0);
        gr.Axis();
        gr.Plot(daty);
        gr.Label('x', "mesh size (2^{n})", 0);
        gr.Label('y', "log(max. error)", 0);
        gr.Title("Maximum Interpolation Error");
        gr.WriteFrame("plots/gradedMesh-logMaxInterpolationError-varN.eps");
    }

    return 0;
}

