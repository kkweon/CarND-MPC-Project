#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"

using namespace std;

struct Solution {
  vector<double> xs;
  vector<double> ys;
  vector<double> psis;
  vector<double> vs;
  vector<double> ctes;
  vector<double> epsis;
  vector<double> deltas;
  vector<double> accs;
};

class MPC {
 public:
  double delta_prev = 0;
  double a_prev = 0.1;
  const int LATENCY_IND = 2;

  MPC();

  virtual ~MPC();

  /**
   * Solve the model given an initial state and polynomial coefficients
   *
   * @param state
   * @param coeffs
   *
   * @return first actuation
   */
  Solution Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs);

};

#endif /* MPC_H */
