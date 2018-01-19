#include <vector>

using namespace std;

//Evaluate the Chebyshev polynomials up to order $n$ in $x$
vector<double> chebPolyEval(const int &n,const double &x) {
    vector<double> V= {1,x};
    for (int k=1; k<n; k++)
        V.push_back(2*x*V[k]-V[k-1]);
    return V;
}
