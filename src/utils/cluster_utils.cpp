#include "cluster_utils.hpp"

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "proto_utils.hpp"
#include "../../lib/progressbar/progressbar.hpp"

//! \param coll Collector containing the algorithm chain
//! \return     Index of the iteration containing the best estimate
Eigen::VectorXd bayesmix::cluster_estimate(
    const Eigen::MatrixXd &alloc_chain) {
  // Initialize objects
  unsigned n_iter = alloc_chain.rows();
  unsigned int n_data = alloc_chain.cols();
  Eigen::MatrixXd mean_diss = Eigen::MatrixXd::Zero(n_data, n_data);
  std::vector<Eigen::SparseMatrix<double> > all_diss;
  progresscpp::ProgressBar bar(n_iter, 60);
  std::cout << "Computing mean dissimilarity... " << std::flush;

  // Loop over pairs (i,j) of data points
  for (int i = 0; i < n_data; i++) {
    for (int j = 0; j < i; j++) {
      // Compute mean dissimilarity for (i,j) by looping over iterations
      for (int k = 0; k < n_iter; k++) {
        if (alloc_chain(k, i) == alloc_chain(k, j)) {
          mean_diss(i, j) += 1.0;
        }
      mean_diss(i, j) = mean_diss(i, j) / n_iter;
      }
    }
  }
  std::cout << "Done" << std::endl;
  std::cout << "Computing cluster estimate..." << std::endl;

  // Compute Frobenius norm error of all iterations
  Eigen::VectorXd errors(n_iter);
  for (int k = 0; k < n_iter; k++) {
    for (int i = 0; i < n_data; i++) {
      for (int j = 0; j < i; j++) {
        int x = (alloc_chain(k, i) == alloc_chain(k, j));
        errors(k) += (x - mean_diss(i, j)) * (x - mean_diss(i, j));
      }
    }
    // Progress bar
    ++bar;
    bar.display();
  }
  bar.done();

  // Find iteration with the least error
  std::ptrdiff_t ibest;
  unsigned int min_err = errors.minCoeff(&ibest);
  std::cout << "Done" << std::endl;
  return alloc_chain.row(ibest).transpose();
}
