#include "newtonIpol.hpp"
#include <mgl2/mgl.h>
#include <cmath>
#include <iostream>


// TODO: Implement the adaptive strategy using greedy algorithm
template <class Function>
void adaptivepolyintp(const Function& f, const double a, const double b,
                      const double tol, const unsigned N,
                      Eigen::VectorXd& adaptive_nodes,
                      Eigen::VectorXd& error_vs_step_no) {
    // Generate sampling points and evaluate $f$ there
    Eigen::VectorXd sampling_points = Eigen::VectorXd::LinSpaced(N, a, b);
    Eigen::VectorXd fvals_at_sampling_points = sampling_points.unaryExpr(f);

    // TODO

}


int main() {
  // Test the interpolant
  VectorXd t(3), y(3), coeffs;
  t << -1, 0, 1;
  y << 2, -4, 6;
  coeffs = divDiff(t, y);
  std::cout << coeffs.transpose() << std::endl; // Expected output: [2 -6 8] 
  
  // TODO

  return 0;
}

