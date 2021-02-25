#include "logit_sb_mixing.h"

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <numeric>
#include <stan/math/prim.hpp>
#include <vector>

#include "mixing_prior.pb.h"
#include "mixing_state.pb.h"
#include "src/utils/proto_utils.h"
#include "src/utils/rng.h"

void LogitSBMixing::initialize() {
  if (prior == nullptr) {
    throw std::invalid_argument("Mixing prior was not provided");
  }
  auto priorcast = cast_prior();
  num_components = priorcast->num_components();
  initialize_state();
  acceptance_rates = Eigen::VectorXd::Zero(num_components);
  n_iter = 0;
}

void LogitSBMixing::initialize_state() {
  auto priorcast = cast_prior();
  if (priorcast->has_normal_prior()) {
    Eigen::VectorXd prior_vec =
        bayesmix::to_eigen(priorcast->normal_prior().mean());
    dim = prior_vec.size();
    state.precision = stan::math::inverse_spd(
        bayesmix::to_eigen(priorcast->normal_prior().var()));
    if (dim != state.precision.cols()) {
      throw std::invalid_argument(
          "Hyperparameters dimensions are not consisent");
    }
    if (priorcast->step_size() <= 0) {
      throw std::invalid_argument("Step size parameter must be > 0");
    }

    state.regression_coeffs = Eigen::MatrixXd(dim, num_components - 1);
    for (int i = 0; i < num_components - 1; i++) {
      state.regression_coeffs.col(i) = prior_vec;
    }

  } else {
    throw std::invalid_argument("Unrecognized mixing prior");
  }
}

Eigen::VectorXd LogitSBMixing::grad_log_full_cond(
    const Eigen::VectorXd &alpha, const unsigned int clust,
    const std::vector<unsigned int> &allocations) {
  auto priorcast = cast_prior();
  Eigen::VectorXd prior_mean =
      bayesmix::to_eigen(priorcast->normal_prior().mean());
  Eigen::VectorXd grad = state.precision * (prior_mean - alpha);
  for (int i = 0; i < allocations.size(); i++) {
    if (allocations[i] >= clust) {
      bool is_curr_clus = (allocations[i] == clust);
      double prob = sigmoid(covariates_ptr->row(i).dot(alpha));
      grad += (is_curr_clus - prob) * covariates_ptr->row(i);
    }
  }
  return grad;
}

double LogitSBMixing::log_like(const Eigen::VectorXd &alpha,
                               const unsigned int clust,
                               const std::vector<unsigned int> &allocations) {
  double like = 0.0;
  for (int i = 0; i < allocations.size(); i++) {
    if (allocations[i] >= clust) {
      bool is_curr_clus = (allocations[i] == clust);
      double prob = sigmoid(covariates_ptr->row(i).dot(alpha));
      like += is_curr_clus * std::log(prob) +
              (1.0 - is_curr_clus) * std::log(1.0 - prob);
    }
  }
  return like;
}

void LogitSBMixing::update_state(
    const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
    const std::vector<unsigned int> &allocations) {
  n_iter += 1;
  // Langevin-Adjusted Metropolis-Hastings step
  unsigned int n = allocations.size();
  auto &rng = bayesmix::Rng::Instance().get();
  auto priorcast = cast_prior();
  Eigen::VectorXd prior_mean =
      bayesmix::to_eigen(priorcast->normal_prior().mean());
  double step = priorcast->step_size();
  double prop_var = std::sqrt(2.0 * step);
  // Loop over clusters, but last alpha is ignored
  for (int h = 0; h < unique_values.size(); h++) {  // TODO -1?
    Eigen::VectorXd state_c = state.regression_coeffs.col(h);
    // Draw proposed state from its distribution
    Eigen::VectorXd prop_mean =
        state_c + step * grad_log_full_cond(state_c, h, allocations);
    auto prop_covar = prop_var * Eigen::MatrixXd::Identity(dim, dim);
    Eigen::VectorXd state_prop =
        stan::math::multi_normal_rng(prop_mean, prop_covar, rng);
    // Compute acceptance ratio
    double prior_ratio =
        -0.5 * ((state_prop - prior_mean).transpose() * state.precision *
                    (state_prop - prior_mean) -
                (state_c - prior_mean).transpose() * state.precision *
                    (state_c - prior_mean))(0);
    double like_ratio = log_like(state_prop, h, allocations) -
                        log_like(state_c, h, allocations);
    double prop_ratio = (-0.5 / prop_var) *
                        ((state_prop - prop_mean).dot(state_prop - prop_mean) -
                         (state_c - prop_mean).dot(state_c - prop_mean));
    double log_accept_ratio = prior_ratio + like_ratio - prop_ratio;
    // Accept with probability ratio
    double p = stan::math::uniform_rng(0.0, 1.0, rng);
    if (p < std::exp(log_accept_ratio)) {
      state.regression_coeffs.col(h) = state_prop;
      acceptance_rates(h) += 1;
    }
  }
}

void LogitSBMixing::set_state_from_proto(
    const google::protobuf::Message &state_) {
  auto &statecast =
      google::protobuf::internal::down_cast<const bayesmix::LogSBState &>(
          state_);
  state.regression_coeffs = bayesmix::to_eigen(statecast.regression_coeffs());
}

void LogitSBMixing::write_state_to_proto(
    google::protobuf::Message *out) const {
  bayesmix::LogSBState state_;
  bayesmix::to_proto(state.regression_coeffs,
                     state_.mutable_regression_coeffs());
  google::protobuf::internal::down_cast<bayesmix::LogSBState *>(out)->CopyFrom(
      state_);
}

Eigen::VectorXd LogitSBMixing::get_weights(
    const Eigen::VectorXd &covariate /*= Eigen::VectorXd(0)*/) const {
  // Compute eta
  std::vector<double> eta(num_components - 1);
  for (int h = 0; h < num_components - 1; h++) {
    eta[h] = covariate.dot(state.regression_coeffs.col(h));
  }
  // Compute cumulative products
  std::vector<double> cumprod(num_components, 1.0);
  for (int h = 1; h < num_components; h++) {
    cumprod[h] = cumprod[h - 1] * sigmoid(-eta[h - 1]);
  }
  // Compute weights
  Eigen::VectorXd weights(num_components);
  for (int h = 0; h < num_components - 1; h++) {
    weights(h) = sigmoid(eta[h]) * cumprod[h];
  }
  weights(num_components - 1) =
      1.0 - std::accumulate(weights.data(),
                            weights.data() + num_components - 1, 0.0);
  return weights;
}
