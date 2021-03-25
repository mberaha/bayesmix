#ifndef SRC_UTILS_DIRICHLET_PROCESS
#define SRC_UTILS_DIRICHLET_PROCESS

#include <algorithm>

#include "src/hierarchies/abstract_hierarchy.h"

class DirichletProcess {
 protected:
  double totalmass;
  std::shared_ptr<AbstractHierarchy> master_hier;
  std::vector<std::shared_ptr<AbstractHierarchy>> unique_values;
  std::vector<int> cards;
  Eigen::VectorXd fullcond_weigths;
  std::vector<double> prior_stickbreak_weights;
  std::vector<double> prior_stickbreak_weights_sums;
  std::vector<std::shared_ptr<AbstractHierarchy>> prior_atoms;
  double sum_stickbreak = 0.0;
  double prev_stick;

 public:
  DirichletProcess() {}
  ~DirichletProcess() {}

  DirichletProcess(std::shared_ptr<AbstractHierarchy> master_hier,
                   double totalmass) {
    this->master_hier = master_hier;
    this->totalmass = totalmass;
    prior_atoms.clear();
    prior_stickbreak_weights.clear();
    prior_stickbreak_weights_sums.clear();
  }

  void set_unique_vals_and_cards(
      std::vector<std::shared_ptr<AbstractHierarchy>> unique_values,
      std::vector<int> cards) {
    this->cards = cards;
    this->unique_values = unique_values;
  }

  void simulate_full_conditional() {
    auto& rng = bayesmix::Rng::Instance().get();
    Eigen::VectorXd dir_params(cards.size() + 1);
    dir_params(0) = totalmass;
    for (int i = 0; i < cards.size(); i++) {
      dir_params(i + 1) = cards[i];
    }

    fullcond_weigths = stan::math::dirichlet_rng(dir_params, rng);
  }

  std::shared_ptr<AbstractHierarchy> draw() {
    auto& rng = bayesmix::Rng::Instance().get();
    int comp = bayesmix::categorical_rng(fullcond_weigths, rng);
    if (comp > 0) {
      auto out = unique_values[comp - 1]->clone();
      return out;
    }

    return sample_prior_retrospective();
  }

  std::shared_ptr<AbstractHierarchy> sample_prior_retrospective() {
    auto& rng = bayesmix::Rng::Instance().get();
    double slice = stan::math::uniform_rng(0, 1, rng);

    while (slice > sum_stickbreak) {
      double stick = stan::math::beta_rng(1, totalmass, rng);
      auto curr_atom = master_hier->clone();
      curr_atom->sample_prior();
      prior_atoms.push_back(curr_atom);
      if (prior_stickbreak_weights.size() == 0) {
        prior_stickbreak_weights.push_back(stick);
        sum_stickbreak = stick;
        prior_stickbreak_weights_sums.push_back(0);
        prior_stickbreak_weights_sums.push_back(stick);
      } else {
        int last = prior_stickbreak_weights.size() - 1;
        double curr_weight = stick * prior_stickbreak_weights[last] *
                             (1 - prev_stick) / prev_stick;
        prior_stickbreak_weights.push_back(curr_weight);
        sum_stickbreak += curr_weight;
        prior_stickbreak_weights_sums.push_back(sum_stickbreak);
      }
      prev_stick = stick;
    }
    auto iter = std::upper_bound(prior_stickbreak_weights_sums.begin(),
                                 prior_stickbreak_weights_sums.end(), slice,
                                 [](double a, double b) { return a >= b; });
    int comp;
    if (iter == prior_stickbreak_weights_sums.end()) {
      comp = prior_atoms.size() - 1;
    } else {
      comp = std::distance(prior_stickbreak_weights_sums.begin(), iter);
    }
    // std::cout << "prior_atoms: " << prior_atoms.size()
    //           << ", prior_stickbreak_weights: "
    //           << prior_stickbreak_weights.size()
    //           << ", prior_stickbreak_weights_sums: "
    //           << prior_stickbreak_weights_sums.size() << ", comp: " << comp
    //           << std::endl;
    return prior_atoms[comp];
  }

  Eigen::VectorXd simulate_stickbreak_threshold(double totalmass, double err) {
    auto& rng = bayesmix::Rng::Instance().get();

    std::vector<double> weights;
    double sticks_cumprod;

    weights.push_back(stan::math::beta_rng(1, totalmass, rng));
    sticks_cumprod = (1 - weights[0]);
    double weight_sum = weights[0];

    while (weight_sum < 1 - err) {
      double stick = stan::math::beta_rng(1, totalmass, rng);
      double weight = stick * sticks_cumprod;
      weights.push_back(weight);
      sticks_cumprod *= (1 - stick);
      weight_sum += weight;
    }
    // weights.push_back(1 - weight_sum);

    Eigen::VectorXd out =
        Eigen::Map<Eigen::VectorXd>(weights.data(), weights.size());
    return out;
  }
};

#endif