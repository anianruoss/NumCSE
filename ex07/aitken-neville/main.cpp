#include <eigen3/Eigen/Dense>
#include <iostream>


using namespace Eigen;

/*!
 * \brief derivIpolEvalAN
 * \param t
 * \param y
 * \param x
 * \return
 */
double derivIpolEvalAN(const VectorXd & t,
                       const VectorXd & y,
                       const double x) {
    assert(t.size() == y.size());

    VectorXd p(y);
    VectorXd dP = VectorXd::Zero(y.size());

    for (int i = 0; i < y.size(); ++i) {
        for (int k = i-1; k >= 0; --k) {
            double s = (x-t(k))*dP(k+1) - (x-t(i))*dP(k) + p(k+1) - p(k);
			dP(k) = s / (t(i) - t(k)); 
            p(k) = p(k+1) + (p(k+1) - p(k))*(x - t(i))/(t(i) - t(k));
        }
    }

    return dP(0);
}


int main() {

    VectorXd t(3), y(3);
    t << -1, 0, 1;
    y << 2, -4, 6;

    double x = 0.5;
    std::cout<< "Polynomial derivative @ x = " << x
             << " is: "<< derivIpolEvalAN(t,y,x) << std::endl;

    return 0;
}

