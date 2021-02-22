#include "lin_reg_uni_hierarchy.h"

#include <Eigen/Dense>
#include <stan/math/prim/err.hpp>

#include "src/utils/eigen_utils.h"
#include "src/utils/proto_utils.h"
#include "src/utils/rng.h"

double LinRegUniHierarchy::like_lpdf(
    const Eigen::RowVectorXd &datum,
    const Eigen::RowVectorXd &covariate /*= Eigen::VectorXd(0)*/) const {
  return stan::math::normal_lpdf(
      datum(0), state.regression_coeffs.dot(covariate), sqrt(state.var));
}

double LinRegUniHierarchy::marg_lpdf(
    const LinRegUni::Hyperparams &params, const Eigen::RowVectorXd &datum,
    const Eigen::RowVectorXd &covariate /*= Eigen::VectorXd(0)*/) const {
  double sig_n = sqrt(
      (1 + (covariate * params.var_scaling_inv * covariate.transpose())(0)) *
      params.scale / params.shape);
  return stan::math::student_t_lpdf(datum(0), 2 * params.shape,
                                    covariate.dot(params.mean), sig_n);
}

LinRegUni::State LinRegUniHierarchy::draw(
    const LinRegUni::Hyperparams &params) {
  auto &rng = bayesmix::Rng::Instance().get();
  LinRegUni::State out;
  out.var = stan::math::inv_gamma_rng(params.shape, params.scale, rng);
  out.regression_coeffs = stan::math::multi_normal_prec_rng(
      params.mean, params.var_scaling / out.var, rng);
  return out;
}

void LinRegUniHierarchy::update_summary_statistics(
    const Eigen::VectorXd &datum, const Eigen::VectorXd &covariate, bool add) {
  if (add) {
    data_sum_squares += datum(0) * datum(0);
    covar_sum_squares += covariate * covariate.transpose();
    mixed_prod += datum(0) * covariate;
  } else {
    data_sum_squares -= datum(0) * datum(0);
    covar_sum_squares -= covariate * covariate.transpose();
    mixed_prod -= datum(0) * covariate;
  }
}

LinRegUni::Hyperparams LinRegUniHierarchy::get_posterior_parameters() {
  if (card == 0) {  // no update possible
    return *hypers;
  }
  // Compute posterior hyperparameters
  LinRegUni::Hyperparams post_params;
  post_params.var_scaling = covar_sum_squares + hypers->var_scaling;
  auto llt = post_params.var_scaling.llt();
  post_params.var_scaling_inv = llt.solve(Eigen::MatrixXd::Identity(dim, dim));
  post_params.mean =
      llt.solve(mixed_prod + hypers->var_scaling * hypers->mean);
  post_params.shape = hypers->shape + 0.5 * card;
  post_params.scale =
      hypers->scale +
      0.5 * (data_sum_squares +
             hypers->mean.transpose() * hypers->var_scaling * hypers->mean -
             post_params.mean.transpose() * post_params.var_scaling *
                 post_params.mean);
  return post_params;
}

void LinRegUniHierarchy::clear_data() {
  mixed_prod = Eigen::VectorXd::Zero(dim);
  data_sum_squares = 0.0;
  covar_sum_squares = Eigen::MatrixXd::Zero(dim, dim);
  card = 0;
  cluster_data_idx = std::set<int>();
}

void LinRegUniHierarchy::initialize_state() {
  state.regression_coeffs = hypers->mean;
  state.var = hypers->scale / (hypers->shape + 1);
}

void LinRegUniHierarchy::initialize_hypers() {
  if (prior->has_fixed_values()) {
    // Set values
    hypers->mean = bayesmix::to_eigen(prior->fixed_values().mean());
    dim = hypers->mean.size();
    hypers->var_scaling =
        bayesmix::to_eigen(prior->fixed_values().var_scaling());
    hypers->var_scaling_inv = stan::math::inverse_spd(hypers->var_scaling);
    hypers->shape = prior->fixed_values().shape();
    hypers->scale = prior->fixed_values().scale();
    // Check validity
    if (dim != hypers->var_scaling.rows()) {
      throw std::invalid_argument(
          "Hyperparameters dimensions are not consistent");
    }
    bayesmix::check_spd(hypers->var_scaling);
    if (hypers->shape <= 0) {
      throw std::invalid_argument("Shape parameter must be > 0");
    }
    if (hypers->scale <= 0) {
      throw std::invalid_argument("Scale parameter must be > 0");
    }
  }

  else {
    throw std::invalid_argument("Unrecognized hierarchy prior");
  }
}

void LinRegUniHierarchy::update_hypers(
    const std::vector<bayesmix::MarginalState::ClusterState> &states) {
  auto &rng = bayesmix::Rng::Instance().get();
  if (prior->has_fixed_values()) {
    return;
  }

  else {
    throw std::invalid_argument("Unrecognized hierarchy prior");
  }
}

void LinRegUniHierarchy::set_state_from_proto(
    const google::protobuf::Message &state_) {
  auto &statecast = google::protobuf::internal::down_cast<
      const bayesmix::MarginalState::ClusterState &>(state_);
  state.regression_coeffs =
      bayesmix::to_eigen(statecast.lin_reg_uni_ls_state().regression_coeffs());
  state.var = statecast.lin_reg_uni_ls_state().var();
  set_card(statecast.cardinality());
}

void LinRegUniHierarchy::write_state_to_proto(
    google::protobuf::Message *out) const {
  bayesmix::LinRegUniLSState state_;
  bayesmix::to_proto(state.regression_coeffs,
                     state_.mutable_regression_coeffs());
  state_.set_var(state.var);

  auto *out_cast = google::protobuf::internal::down_cast<
      bayesmix::MarginalState::ClusterState *>(out);
  out_cast->mutable_lin_reg_uni_ls_state()->CopyFrom(state_);
  out_cast->set_cardinality(card);
}

void LinRegUniHierarchy::write_hypers_to_proto(
    google::protobuf::Message *out) const {
  bayesmix::LinRegUniPrior hypers_;
  bayesmix::to_proto(hypers->mean,
                     hypers_.mutable_fixed_values()->mutable_mean());
  bayesmix::to_proto(hypers->var_scaling,
                     hypers_.mutable_fixed_values()->mutable_var_scaling());
  hypers_.mutable_fixed_values()->set_shape(hypers->shape);
  hypers_.mutable_fixed_values()->set_scale(hypers->scale);

  google::protobuf::internal::down_cast<bayesmix::LinRegUniPrior *>(out)
      ->mutable_fixed_values()
      ->CopyFrom(hypers_.fixed_values());
}
