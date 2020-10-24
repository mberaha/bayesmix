#ifndef NEAL8_HPP
#define NEAL8_HPP

#include "Neal2.hpp"

//! Template class for Neal's algorithm 8 for conjugate hierarchies

//! This class implements Neal's Gibbs sampling algorithm 8 that generates a
//! Markov chain on the clustering of the provided data.
//!
//! It is a generalization of Neal's algorithm 2 which works for any
//! hierarchical model, even non-conjugate ones, unlike its predecessor. The
//! main difference is the presence of a fixed number of additional unique
//! values, called the auxiliary blocks, which are constantly updated and from
//! which new clusters choose their unique values. They offset the lack of
//! conjugacy of the model by allowing an estimate of the (uncomputable)
//! marginal density via a weighted mean on these new blocks. Other than this
//! and some minor adjustments in the allocation sampling phase to circumvent
//! non-conjugacy, it is the same as Neal's algorithm 2.

class Neal8 : public Neal2 {
 protected:
  //! Number of auxiliary blocks
  unsigned int n_aux = 3;

  //! Vector of auxiliary blocks
  std::vector<std::shared_ptr<HierarchyBase>> aux_unique_values;

  // AUXILIARY TOOLS
  //! Computes marginal contribution of a given iteration & cluster
  Eigen::VectorXd density_marginal_component(
      std::shared_ptr<HierarchyBase> temp_hier) override;

  // ALGORITHM FUNCTIONS
  void print_startup_message() const override;
  void sample_allocations() override;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~Neal8() = default;
  Neal8() = default;

  // GETTERS AND SETTERS
  unsigned int get_n_aux() const { return n_aux; }
  void set_n_aux(const unsigned int n_aux_) override {
    n_aux = n_aux_;
    // Rebuild the correct amount of auxiliary blocks
    aux_unique_values.clear();
    for (size_t i = 0; i < n_aux; i++) {
      aux_unique_values.push_back(unique_values[0]);
    }
  }

  // void initalize() { // TODO
  //   // ...
  //   // Initialize auxiliary blocks
  //   for (size_t i = 0; i < n_aux; i++) {
  //     aux_unique_values.push_back(unique_values[0]);
  //   }
  // }

  void print_id() const override { std::cout << "N8" << std::endl; }  // TODO
};

#endif  // NEAL8_HPP
