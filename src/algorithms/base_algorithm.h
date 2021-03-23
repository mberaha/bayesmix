#ifndef BAYESMIX_ALGORITHMS_BASE_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_BASE_ALGORITHM_H_

#include <lib/progressbar/progressbar.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "algorithm_id.pb.h"
#include "algorithm_params.pb.h"
#include "algorithm_state.pb.h"
#include "src/collectors/base_collector.h"
#include "src/hierarchies/base_hierarchy.h"
#include "src/mixings/base_mixing.h"

//! Abstract template class for a Gibbs sampling iterative BNP algorithm.

//! This template class implements a generic algorithm that generates a Markov
//! chain on the clustering of the provided data.
//!
//! An algorithm that inherits from this abstract class will have multiple
//! iterations of the same step. Steps are further split into substeps, each of
//! which updates specific values of the state of the Markov chain, which is
//! composed of an allocations vector and a unique values vector (see below).
//! This is known as a Gibbs sampling structure, where a set of values is
//! updated according to a conditional distribution given all other values.
//! The underlying model for the data is assumed to be a so-called hierarchical
//! model, where each datum is independently drawn from a common likelihood
//! function, whose parameters are specific to each unit and are iid generated
//! from a random probability measure, called mixture. Different data points
//! may have the same parameters as each other, and thus a clustering structure
//! on data emerges, with each cluster being identified by its own parameters,
//! called unique values. These will often be generated from the centering
//! distribution, which is the expected value of the mixture, or from its
//! posterior update. The allocation of a datum is instead the label that
//! indicates the cluster it is currently assigned to. The probability
//! distribution for data from each cluster is called a hierarchy and it can
//! have its own hyperparameters, either random themselves or fixed. The model
//! therefore is:
//!   x_i ~ f(x_i|phi_(c_i))  (data likelihood);
//! phi_c ~ G                 (unique values distribution);
//!     G ~ MM                (mixture model);
//!  E[G] = G0                (centering distribution),
//! where c_i is the allocation of the i-th datum.
//!
//! This class is templatized over the types of the elements of this model: the
//! hierarchies of cluster, their hyperparameters, and the mixing mode.

class BaseAlgorithm {
 protected:
  // METHOD PARAMETERS
  //! Iterations of the algorithm
  unsigned int maxiter = 1000;
  //! Number of burn-in iterations, which will be discarded
  unsigned int burnin = 100;

  // DATA AND VALUES CONTAINERS
  //! Initial number of clusters, only used for initialization
  unsigned int init_num_clusters = 0;
  //! Matrix of row-vectorial data points
  Eigen::MatrixXd data;
  //! Vector of allocation labels for each datum
  std::vector<unsigned int> allocations;
  //! Vector of pointers to Hierarchy objects that identify each cluster
  std::vector<std::shared_ptr<AbstractHierarchy>> unique_values;
  //! Covariates matrix for the Hierarchy objects
  Eigen::MatrixXd hier_covariates;
  //! Pointer to the Mixing object
  std::shared_ptr<BaseMixing> mixing;
  //! Covariates matrix for the Mixing object
  Eigen::MatrixXd mix_covariates;
  //!
  bayesmix::AlgorithmState curr_state;
  //!
  virtual bool update_hierarchy_params() { return false; }

  // AUXILIARY TOOLS
  //! Returns the values of an algo iteration as a Protobuf object
  bayesmix::AlgorithmState get_state_as_proto(unsigned int iter);
  bool update_state_from_collector(BaseCollector *coll);

  // ALGORITHM FUNCTIONS
  //!
  virtual void initialize();
  //!
  virtual void print_startup_message() const = 0;
  //!
  virtual void sample_allocations() = 0;
  //!
  virtual void sample_unique_values() = 0;
  //!
  void update_hierarchy_hypers();
  //!
  virtual void print_ending_message() const {
    std::cout << "Done" << std::endl;
  };

  //! Saves the current iteration's state in Protobuf form to a collector
  void save_state(BaseCollector *collector, unsigned int iter) {
    collector->collect(get_state_as_proto(iter));
  }

  //! Single step of algorithm
  virtual void step() {
    sample_allocations();
    sample_unique_values();
    update_hierarchy_hypers();
    mixing->update_state(unique_values, allocations);
  }

 public:
  //!
  virtual bool requires_conjugate_hierarchy() const { return false; }
  //!
  virtual bool is_conditional() const = 0;

  //! Runs the algorithm and saves the whole chain to a collector
  void run(BaseCollector *collector, bool log = true) {
    initialize();
    print_startup_message();
    unsigned int iter = 0;
    collector->start_collecting();
    progresscpp::ProgressBar bar(maxiter, 60);

    while (iter < maxiter) {
      step();
      if (iter >= burnin) {
        save_state(collector, iter);
      }
      iter++;

      if (log) {
        ++bar;
        bar.display();
      }
    }
    collector->finish_collecting();
    bar.done();
    print_ending_message();
  }

  // ESTIMATE FUNCTION
  virtual Eigen::VectorXd lpdf_from_state(
      const Eigen::MatrixXd &grid, const Eigen::RowVectorXd &hier_covariate,
      const Eigen::RowVectorXd &mix_covariate) = 0;
  //! Evaluates the logpdf for each single iteration on a given grid of points
  virtual Eigen::MatrixXd eval_lpdf(
      BaseCollector *const collector, const Eigen::MatrixXd &grid,
      const Eigen::RowVectorXd &hier_covariate = Eigen::RowVectorXd(0),
      const Eigen::RowVectorXd &mix_covariate = Eigen::RowVectorXd(0));

  // DESTRUCTOR AND CONSTRUCTORS
  virtual ~BaseAlgorithm() = default;
  BaseAlgorithm() = default;

  // GETTERS AND SETTERS
  unsigned int get_maxiter() const { return maxiter; }
  unsigned int get_burnin() const { return burnin; }

  void set_maxiter(const unsigned int maxiter_) { maxiter = maxiter_; }
  void set_burnin(const unsigned int burnin_) { burnin = burnin_; }
  void set_init_num_clusters(const unsigned int init_) {
    init_num_clusters = init_;
  }
  void set_mixing(const std::shared_ptr<BaseMixing> mixing_) {
    mixing = mixing_;
  }
  void set_data(const Eigen::MatrixXd &data_) { data = data_; }
  void set_hier_covariates(const Eigen::MatrixXd &cov) {
    hier_covariates = cov;
  }
  void set_mix_covariates(const Eigen::MatrixXd &cov) { mix_covariates = cov; }
  void set_hierarchy(const std::shared_ptr<AbstractHierarchy> hier_) {
    unique_values.clear();
    unique_values.push_back(hier_);
  }
  virtual bayesmix::AlgorithmId get_id() const = 0;
  virtual void read_params_from_proto(const bayesmix::AlgorithmParams &params);
};

#endif  // BAYESMIX_ALGORITHMS_BASE_ALGORITHM_H_
