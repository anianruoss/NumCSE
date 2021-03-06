#include <cmath>
#include <iomanip>
#include <iostream>
#include <mgl2/mgl.h>


int main() {
    double *hValues = new double[21];
    double *cancellationErrors = new double[21];
    double *noCancellationErrors = new double[21];

    double value = 1.2;
    double h = 1;

    int sep = 20;

    std::cout << std::setw(sep) << "h"
              << std::setw(sep) << "exact"
              << std::setw(sep) << "cancellation"
              << std::setw(sep) << "error"
              << std::setw(sep) << "improved"
              << std::setw(sep) << "error"
              << std::endl;

    std::cout << std::fixed << std::scientific;

    for (int i = 0; i < 21; ++i) {
        double exact_value = std::cos(value);
        double cancellation = (std::sin(value+h) - std::sin(value))/h;
        double no_cancellation = 2.0*std::cos(value+h/2.0)*std::sin(h/2.0)/h;

        hValues[i] = h;
        cancellationErrors[i] = std::abs(cancellation - exact_value);
        cancellationErrors[i] /= exact_value;
        noCancellationErrors[i] = std::abs(no_cancellation - exact_value);
        noCancellationErrors[i] /= exact_value;

        std::cout << std::setw(sep) << h
                  << std::setw(sep) << exact_value
                  << std::setw(sep) << cancellation
                  << std::setw(sep) << cancellationErrors[i]
                  << std::setw(sep) << no_cancellation
                  << std::setw(sep) << noCancellationErrors[i]
                  << std::endl;
        h /= 10;
    }

    mglData stepSize, dataRef;
    stepSize.Link(hValues, 21);
    dataRef.Link(hValues, 21);

    mglData data1, data2;
    data1.Link(cancellationErrors, 21);
    data2.Link(noCancellationErrors, 21);

    mglGraph *gr = new mglGraph;
    gr->Title("Error of approximation of f'(x_0)");
    gr->SetRanges(hValues[20],hValues[0],1e-20,1.5);
    gr->Label('x', "h",0);
    gr->Label('y', "Relative Error",0);
    gr->SetFunc("lg(x)","lg(y)");

    double xTicks[] = {1e-20,1e-15,1e-10,1e-5,1e+0};
    double yTicks[] = {1e-20,1e-15,1e-10,1e-5,1e+0};
    gr->SetTicksVal('x', mglData(5,xTicks),
                    "10^{-20}\n10^{-15}\n10^{-10}\n\\10^{-5}\n\\10^{0}");
    gr->SetTicksVal('y', mglData(5,yTicks),
                    "10^{-20}\n10^{-15}\n10^{-10}\n\\10^{-5}\n\\10^{0}");
    gr->Axis();

    gr->Plot(stepSize,data1,"k+");
    gr->AddLegend("cancellation", "k+");
    gr->Plot(stepSize,data2,"r +");
    gr->AddLegend("no cancellation", "r +");
    gr->Plot(stepSize,dataRef,"b|");
    gr->AddLegend("O(h)","b|");
    gr->Legend(1);
    gr->WriteFrame("plots/avoid-cancellation.eps");

    return 0;
}

