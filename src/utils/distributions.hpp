#ifndef DISTRIBUTIONS_HPP
#define DISTRIBUTIONS_HPP

#include <Eigen/Dense>
#include <random>
#include <stan/math/prim/prob.hpp>

namespace bayesmix {
int categorical_rng(const Eigen::VectorXd &probas, std::mt19937_64 &rng,
                    int start = 0);
double multi_normal_prec_lpdf(const Eigen::VectorXd &datum,
                              const Eigen::VectorXd &mean,
                              const Eigen::MatrixXd &prec_chol,
                              double prec_logdet);
}  // namespace bayesmix

#endif  // DISTRIBUTIONS_HPP
