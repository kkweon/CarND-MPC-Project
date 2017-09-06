#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

#define DEBUG(element, string) \
  if (1) std::cout << (string) << ": " << (element) << std::endl

using CppAD::AD;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

// Set the timestep length and duration
const size_t N  = 12;
const double dt = 0.05;

const size_t STATE_SIZE    = 6;
const size_t ACTUATOR_SIZE = 2;

const double MAX_REAL_NUMBER = 1.0e19;
const double MAX_DELTA       = 0.436332;
const double MAX_ACC         = 1.0;

// State: [x, y, psi, v, cte, epsi]
// Actuation: [delta, a]
const size_t x_start     = 0;
const size_t y_start     = x_start + N;
const size_t psi_start   = y_start + N;
const size_t v_start     = psi_start + N;
const size_t cte_start   = v_start + N;
const size_t epsi_start  = cte_start + N;
const size_t delta_start = epsi_start + N;
const size_t a_start     = delta_start + N - 1;

// Reference velocity
const double ref_v = 85;

namespace {

AD<double> polyeval(const Eigen::VectorXd& coeffs, const AD<double>& x) {
  AD<double> result;
  for (auto i = 0; i < coeffs.size(); ++i) {
    result += coeffs[i] * CppAD::pow(x, i);
  }
  return result;
}

AD<double> polyprimeeval(const Eigen::VectorXd& coeffs, const AD<double>& x) {
  AD<double> result;
  for (auto i = 1; i < coeffs.size(); ++i) {
    result += coeffs[i] * CppAD::pow(x, i - 1) * i;
  }
  return result;
}
}

class FG_eval {
 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;
  vector<double> prev_actuations;
  FG_eval(Eigen::VectorXd coeffs, vector<double> prev_actuations) {
    this->coeffs          = coeffs;
    this->prev_actuations = prev_actuations;
  }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;

  /**
   *
   * Called by ipopt internally
   *
   * @param fg cost constraints
   * @param vars variables (state & actuation)
   *
   */
  void operator()(ADvector& fg, const ADvector& vars) {
    fg[0] = 0;

    // Main Cost
    for (size_t t = 0; t < N; ++t) {
      fg[0] += CppAD::pow(vars[cte_start + t], 2);
      fg[0] += CppAD::pow(vars[epsi_start + t], 2);
      fg[0] += CppAD::pow(vars[v_start + t] - ref_v, 2);
    }

    // Minimize the use of actuators.
    for (size_t t = 0; t < N - 1; ++t) {
      fg[0] += CppAD::pow(vars[delta_start + t], 2);
      fg[0] += 10 * CppAD::pow(vars[a_start + t], 2);
    }

    for (size_t t = 0; t < N - 2; ++t) {
      fg[0] += 1000 * CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2);
      fg[0] += CppAD::pow(vars[a_start + t + 1] - vars[a_start + t], 2);
    }

    // Set up constraints
    fg[1 + x_start]    = vars[x_start];
    fg[1 + y_start]    = vars[y_start];
    fg[1 + psi_start]  = vars[psi_start];
    fg[1 + v_start]    = vars[v_start];
    fg[1 + cte_start]  = vars[cte_start];
    fg[1 + epsi_start] = vars[epsi_start];

    for (size_t t = 1; t < N; ++t) {
      // state at time t
      AD<double> x0    = vars[x_start + t - 1];
      AD<double> y0    = vars[y_start + t - 1];
      AD<double> psi0  = vars[psi_start + t - 1];
      AD<double> v0    = vars[v_start + t - 1];
      AD<double> cte0  = vars[cte_start + t - 1];
      AD<double> epsi0 = vars[epsi_start + t - 1];

      // state at time t + 1
      AD<double> x1    = vars[x_start + t];
      AD<double> y1    = vars[y_start + t];
      AD<double> psi1  = vars[psi_start + t];
      AD<double> v1    = vars[v_start + t];
      AD<double> cte1  = vars[cte_start + t];
      AD<double> epsi1 = vars[epsi_start + t];

      // actuation at time t
      AD<double> delta0  = vars[delta_start + t - 1];
      AD<double> a0      = vars[a_start + t - 1];
      AD<double> f0      = polyeval(coeffs, x0);
      AD<double> psides0 = CppAD::atan(polyprimeeval(coeffs, x0));

      // Here's `x` to get you started.
      // The idea here is to constraint this value to be 0.
      //
      // Recall the equations for the model:
      // x_[t+1] = x[t] + v[t] * cos(psi[t]) * dt
      // y_[t+1] = y[t] + v[t] * sin(psi[t]) * dt
      // psi_[t+1] = psi[t] + v[t] / Lf * delta[t] * dt
      // v_[t+1] = v[t] + a[t] * dt
      // cte[t+1] = f(x[t]) - y[t] + v[t] * sin(epsi[t]) * dt
      // epsi[t+1] = psi[t] - psides[t] + v[t] * delta[t] / Lf * dt
      fg[1 + x_start + t]    = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[1 + y_start + t]    = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[1 + psi_start + t]  = psi1 - (psi0 + v0 * delta0 / Lf * dt);
      fg[1 + v_start + t]    = v1 - (v0 + a0 * dt);
      fg[1 + cte_start + t]  = cte1 - (f0 - y0 + v0 * CppAD::sin(epsi0) * dt);
      fg[1 + epsi_start + t] = epsi1 - (psi0 - psides0 + v0 * delta0 / Lf * dt);
    }
  }
};

//
// MPC class definition implementation.
//
MPC::MPC()  = default;
MPC::~MPC() = default;

Solution MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  double x    = state[0];
  double y    = state[1];
  double psi  = state[2];
  double v    = state[3];
  double cte  = state[4];
  double epsi = state[5];

  // Set the number of model variables (includes both states and inputs).
  size_t n_vars = N * STATE_SIZE + (N - 1) * ACTUATOR_SIZE;
  // Set the number of constraints
  size_t n_constraints = N * STATE_SIZE;

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (size_t i = 0; i < n_vars; i++) {
    vars[i] = 0;
  }
  vars[x_start]    = x;
  vars[y_start]    = y;
  vars[psi_start]  = psi;
  vars[v_start]    = v;
  vars[cte_start]  = cte;
  vars[epsi_start] = epsi;

  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);

  // Set lowerbound & upperbound for variables
  for (size_t i = 0; i < delta_start; ++i) {
    vars_lowerbound[i] = -MAX_REAL_NUMBER;
    vars_upperbound[i] = MAX_REAL_NUMBER;
  }
  // Set lowerbound & upperbound for delta [-25degree, 25degree]
  for (size_t i = delta_start; i < a_start; ++i) {
    vars_lowerbound[i] = -MAX_DELTA;
    vars_upperbound[i] = MAX_DELTA;
  }
  // Fix previous delta since it's in the past
  for (size_t i = delta_start; i < delta_start + LATENCY_IND; ++i) {
    vars_lowerbound[i] = delta_prev;
    vars_upperbound[i] = delta_prev;
  }
  // Do the same for Accelerations
  for (size_t i = a_start; i < n_vars; ++i) {
    vars_lowerbound[i] = -MAX_ACC;
    vars_upperbound[i] = MAX_ACC;
  }

  for (size_t i = a_start; i < a_start + LATENCY_IND; ++i) {
    vars_lowerbound[i] = a_prev;
    vars_upperbound[i] = a_prev;
  }

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (size_t i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }

  constraints_lowerbound[x_start]    = x;
  constraints_lowerbound[y_start]    = y;
  constraints_lowerbound[psi_start]  = psi;
  constraints_lowerbound[v_start]    = v;
  constraints_lowerbound[cte_start]  = cte;
  constraints_lowerbound[epsi_start] = epsi;

  constraints_upperbound[x_start]    = x;
  constraints_upperbound[y_start]    = y;
  constraints_upperbound[psi_start]  = psi;
  constraints_upperbound[v_start]    = v;
  constraints_upperbound[cte_start]  = cte;
  constraints_upperbound[epsi_start] = epsi;

  // object that computes objective and constraints
  FG_eval fg_eval(coeffs, {delta_prev, a_prev});

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // DEBUG(options.size(), "options.size()");
  // DEBUG(vars.size(), "vars.size()");
  // DEBUG(vars_lowerbound.size(), "vars_lowerbound.size()");
  // DEBUG(vars_upperbound.size(), "vars_upperbound.size()");
  // DEBUG(constraints_lowerbound.size(), "constraints_lowerbound.size()");
  // DEBUG(constraints_upperbound.size(), "constraints_upperbound.size()");

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;

  // Return the first actuator values.
  // The variables can be accessed with `solution.x[i]`.
  Solution sol;

  for (unsigned int i = 0; i < N - 1; ++i) {
    sol.xs.push_back(solution.x[x_start + i]);
    sol.ys.push_back(solution.x[y_start + i]);
    sol.deltas.push_back(solution.x[delta_start + i]);
    sol.accs.push_back(solution.x[a_start + i]);
  }

  // for (auto& elem : sol.deltas) {
  //   DEBUG(elem, "sol.deltas[?]");
  // }

  return sol;
}
