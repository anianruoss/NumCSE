#include <iostream>
#include <eigen3/Eigen/Dense>


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
    // TODO

}


int main() {

    VectorXd t(3), y(3);
    t << -1, 0, 1;
    y << 2, -4, 6;

    double x = 0.5;
    std::cout<< "Polynomial derivative @ x = " << x << " is: "<< derivIpolEvalAN(t,y,x) << std::endl;

    return 0;
}

