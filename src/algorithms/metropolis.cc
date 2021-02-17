#include "metropolis.h"

#include <iostream>
#include <stan/math/prim/prob/multi_normal_rng.hpp>
#include <stan/math/prim/prob/uniform_rng.hpp>

#include "src/utils/rng.h"

Eigen::VectorXd Metropolis::standard_mean() const { return state; }

Eigen::VectorXd Metropolis::mala_mean() const {
  Eigen::VectorXd grad = (-1.0 / true_var) * state;
  for (int i = 0; i < data.rows(); i++) {
    grad += (data(i) - inv_logit(state.dot(covariates.row(i)))) *
            covariates.row(i);
  }
  return state + penal * grad;
}

void Metropolis::metropolis_hastings_step() {
  // Draw proposed state from proposal
  auto &rng = bayesmix::Rng::Instance().get();
  Eigen::VectorXd mean = use_mala ? mala_mean() : standard_mean();
  auto covar = prop_var * Eigen::MatrixXd::Identity(dim, dim);
  Eigen::VectorXd proposed = stan::math::multi_normal_rng(mean, covar, rng);
  double exp1 =
      (-0.5 / true_var) * (proposed.dot(proposed) - state.dot(state));
  double ratio1 = std::exp(exp1);
  double ratio2 = 1.0;
  for (int i = 0; i < data.rows(); i++) {
    double exp2a = data(i) * covariates.row(i).dot(proposed);
    double exp2b = covariates.row(i).dot(proposed);
    ratio2 *= std::exp(exp2a) / (1 + std::exp(exp2b));
  }
  double ratio3 =
      std::exp((-0.5 / prop_var) * ((mean - proposed).dot(mean - proposed) -
                                    (mean - state).dot(mean - state)));
  double ratio = ratio1 * ratio2 * ratio3;
  // Accept with probability ratio
  double p = stan::math::uniform_rng(0.0, 1.0, rng);
  std::cout << ratio << std::endl;
  if (p <= ratio) {
    state = proposed;
  }
}

void Metropolis::output() {
  std::cout << "#" << iter << ":\t";
  for (int i = 0; i < dim; i++) {
    std::cout << state(i) << " ";
  }
  std::cout << std::endl;
}
