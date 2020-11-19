#include "semihdp_sampler.hpp"

#include <proto/cpp/marginal_state.pb.h>

#include <algorithm>
#include <src/utils/eigen_utils.hpp>

SemiHdpSampler::SemiHdpSampler(const std::vector<MatrixXd> &data)
    : data(data) {
  ngroups = data.size();
  n_by_group.resize(ngroups);
  for (int i = 0; i < ngroups; i++) n_by_group[i] = data[i].size();

  master_hierarchy.set_mu0(0);
  master_hierarchy.set_lambda0(0.1);
  master_hierarchy.set_alpha0(2);
  master_hierarchy.set_beta0(2);
}

void SemiHdpSampler::initialize() {
  auto rng = bayesmix::Rng::Instance().get();

  int INIT_N_CLUS = 5;
  VectorXd probas = VectorXd::Ones(INIT_N_CLUS);
  probas /= (1.0 * INIT_N_CLUS);
  c.resize(ngroups);
  s.resize(ngroups);
  v.resize(ngroups);
  t.resize(ngroups);
  theta_star.resize(ngroups);
  theta_tilde.resize(ngroups);
  n_by_theta_star.resize(ngroups);

  for (int l = 0; l < INIT_N_CLUS; l++) {
    NNIGHierarchy hierarchy = master_hierarchy;
    hierarchy.draw();
    taus.push_back(hierarchy);
  }

  for (int i = 0; i < ngroups; i++) {
    c[i] = i;
    s[i].resize(n_by_group[i]);

    for (int j = 0; j < INIT_N_CLUS; j++) s[i][j] = j;
    for (int j = INIT_N_CLUS; j < n_by_group[i]; j++) {
      s[i][j] = bayesmix::categorical_rng(probas, rng);
    }

    v[i].resize(2 * INIT_N_CLUS);
    t[i].resize(2 * INIT_N_CLUS);

    for (int l = 0; l < INIT_N_CLUS; l++) {
      NNIGHierarchy hierarchy = master_hierarchy;
      hierarchy.draw();
      theta_tilde[i].push_back(hierarchy);
      theta_star[i].push_back(hierarchy);
      v[i][l] = l;
      t[i][l] = -1;
    }

    for (int l = 0; l < INIT_N_CLUS; l++) {
      theta_star[i].push_back(taus[l]);
      v[i][INIT_N_CLUS + l] = -1;
      t[i][INIT_N_CLUS + l] = l;
    }
  }
  m = std::vector<int>(0, taus.size());
}

void SemiHdpSampler::update_unique_vals() {
  // Loop through the theta stars, if its equal to some theta_tilde, we
  // update it on the fly. Otherwise we store the the data in a vector
  // and then update all the taus

  std::vector<MatrixXd> data_by_tau(taus.size());
  for (int h = 0; h < taus.size(); h++) {
    data_by_tau[h] = Eigen::MatrixXd(0, 1);
  }

  for (int r = 0; r < ngroups; r++) {
    std::vector<MatrixXd> data_by_theta_star(theta_star[r].size());
    for (int i = 0; i < ngroups; i++) {
      if (c[i] == r) {
        for (int j = 0; j < n_by_group[i]; j++) {
          bayesmix::append_by_row(&data_by_theta_star[s[i][j]],
                                  data[i].row(j));
        }
      }
    }

    for (int l = 0; l < theta_star[r].size(); l++) {
      if (v[r][l] >= 0) {
        if (data_by_theta_star[l].rows() > 0)
          theta_tilde[r][v[r][l]].sample_given_data(data_by_theta_star[l]);
        else
          theta_tilde[r][v[r][l]].draw();
        theta_star[r][l] = theta_tilde[r][v[r][l]];
      } else {
        bayesmix::append_by_row(&data_by_tau[t[r][l]], data_by_theta_star[l]);
      }
    }
  }

  for (int h = 0; h < taus.size(); h++) {
    if (data_by_tau[h].rows() > 0)
      taus[h].sample_given_data(data_by_tau[h]);
    else
      taus[h].draw();
  }

  // reassign stuff to teta_stars
  for (int i = 0; i < ngroups; i++) {
    for (int l = 0; l < theta_star[i].size(); l++) {
      if (t[i][l] >= 0) {
        theta_star[i][l] = taus[t[i][l]];
        // bayesmix::MarginalState::ClusterVal state;
        // taus[t[i][l]].write_state_to_proto(&state);
        // theta_star[i][l].set_state(&state);
      }
    }
  }
}

void SemiHdpSampler::update_s() {
  auto rng = bayesmix::Rng::Instance().get();

  // compute counts
  m = std::vector<int>(taus.size(), 0);
  for (int r = 0; r < ngroups; r++) {
    for (int l = 0; l < t[r].size(); l++)
      if (t[r][l] >= 0) {
        std::cout << "t[r][l]: " << t[r][l] << std::endl;
        m[t[r][l]] += 1;
      }

    n_by_theta_star[r] = std::vector<int>(theta_star[r].size(), 0);
    for (int i = 0; i < ngroups; i++) {
      if (c[i] == r) {
        for (int j = 0; j < n_by_group[i]; j++) {
          n_by_theta_star[r][s[i][j]] += 1;
        }
      }
    }
  }

  int m_sum = std::accumulate(m.begin(), m.end(), 0);

  // cicle through observations
  for (int i = 0; i < ngroups; i++) {
    int r = c[i];
    for (int j = 0; j < n_by_group[i]; j++) {
      int s_old = s[i][j];
      // remove current observation from its allocation
      // TODO: check what happens if n becomes 0?
      n_by_theta_star[r][s[i][j]] -= 1;
      if (t[r][s_old] >= 0) {
        m[t[r][s_old]] -= 1;
      }

      VectorXd probas = VectorXd::Zero(theta_star[r].size() + 1);
      for (int l = 0; l < theta_star[r].size(); l++) {
        std::cout << "l: " << l << ", mean: " << theta_star[r][l].get_mean()
                  << ", sd: " << theta_star[r][l].get_sd();
        double log_n;
        if (n_by_theta_star[r][l] > 0)
          log_n = std::log(1.0 * n_by_theta_star[r][l]);
        else
          log_n = 1e-20;

        std::cout << ", lpdf: " << theta_star[r][l].lpdf(data[i].row(j))
                  << ", log_n: " << log_n << std::endl;
        probas[l] = log_n + theta_star[r][l].lpdf(data[i].row(j));
      }

      std::cout << "probas1: " << probas.transpose() << std::endl;
      double margG0 = std::log(w) + master_hierarchy.marg_lpdf(data[i].row(j));

      VectorXd hdp_contribs(taus.size() + 1);
      for (int h = 0; h < taus.size(); h++) {
        double log_m;
        if (m[h] > 0)
          log_m = std::log(1.0 * m[h]);
        else
          log_m = 1e-20;
        hdp_contribs[h] = log_m + taus[h].lpdf(data[i].row(j));
      }

      hdp_contribs[taus.size()] =
          std::log(gamma) + master_hierarchy.marg_lpdf(data[i].row(j));
      double margHDP = std::log(1 - w) + stan::math::log_sum_exp(hdp_contribs);
      VectorXd marg(2);
      marg << margG0, margHDP;
      std::cout << "marg: " << marg.transpose() << std::endl;
      probas[theta_star[r].size()] =
          std::log(alpha) + stan::math::log_sum_exp(marg);

      std::cout << "probas: " << stan::math::softmax(probas).transpose()
                << std::endl;
      int snew = bayesmix::categorical_rng(stan::math::softmax(probas), rng);
      s[i][j] = snew;
      if (snew < probas.size()) {
        n_by_theta_star[r][snew] += 1;
      } else {
        n_by_theta_star[r].push_back(1);
        if (stan::math::uniform_rng(0, 1, rng) < w) {
          // sample from G0, add it to theta_star and theta_tilde and adjust
          // counts and stuff
          NNIGHierarchy hierarchy = master_hierarchy;
          hierarchy.sample_given_data(data[i].row(j));
          theta_tilde[r].push_back(hierarchy);
          theta_star[r].push_back(hierarchy);
          t[r].push_back(-1);
          v[r].push_back(theta_tilde[r].size() - 1);
          n_by_theta_star[r].push_back(1);
        } else {
          // sample from Gtilde
          v[r].push_back(-1);
          int tnew = bayesmix::categorical_rng(
              stan::math::softmax(hdp_contribs), rng);
          t[r].push_back(tnew);
          if (tnew < taus.size()) {
            theta_star[r].push_back(taus[tnew]);
            m[tnew] += 1;
          } else {
            NNIGHierarchy hierarchy = master_hierarchy;
            hierarchy.sample_given_data(data[i].row(j));
            taus.push_back(hierarchy);
            m.push_back(1);
            theta_star[r].push_back(hierarchy);
          }
        }
      }
    }
  }
}

void SemiHdpSampler::update_t() {}

void SemiHdpSampler::update_c() {}

void SemiHdpSampler::update_w() {
  auto rng = bayesmix::Rng::Instance().get();
  // cnt how many from the iodisincratic
  int cnt = 0;
  int tot = 0;
  for (int r = 0; r < ngroups; r++) {
    if (std::find(c.begin(), c.end(), r) != c.end()) {
      for (auto v : v[r]) {
        tot += 1;
        cnt += 1.0 * (v >= 0);
      }
    }
  }
  w = stan::math::beta_rng(a_w + cnt, b_w + tot - cnt, rng);

}

void SemiHdpSampler::relabel() {
  // find the theta_* which are not allocated
  // need to loop through the restaurants and then trhough each group
  // entering in the restaurant

  // make sure there are no holes in v and t
  int min_t = INT_MAX;
  for (int i = 0; i < ngroups; i++) {
    std::vector<int> v_sorted(v[i]);
    std::sort(v_sorted.begin(), v_sorted.end());

    auto it = std::upper_bound(v_sorted.begin(), v_sorted.end(), -1);
    if (it != v[i].end()) {
      int min_v = *it;
      assert(min_v == 0);
    }

    std::vector<int> t_sorted(t[i]);
    std::sort(t_sorted.begin(), t_sorted.end());

    auto it2 = std::upper_bound(t_sorted.begin(), t_sorted.end(), -1);
    if (it2 != t[i].end()) {
      int min_t_temp = *it2;
      min_t = std::min(min_t, min_t_temp);
    }
  }
  assert(min_t == 0);

  for (int r = 0; r < ngroups; r++) {
    // std::cout << "restaurant: " << r << std::endl;
    std::vector<int> groups;
    std::vector<bool> isused(theta_star[r].size(), false);
    for (int i = 0; i < ngroups; i++) {
      if (c[i] == r) groups.push_back(r);
    }

    for (auto i : groups) {
      for (int j = 0; j < n_by_group[i]; j++) {
        isused[s[i][j]] = true;
      }
    }

    // DEBUGGING
    // std::cout << "restaurant: " << r << ", groups: ";
    // for (auto k : groups) std::cout << k << ", ";
    // std::cout << std::endl;
    // std::cout << "Unused theta_stars: ";
    // for (int l = 0; l < isused.size(); l++) {
    //   if (!isused[l]) std::cout << l << ", ";
    // }
    // std::cout << std::endl;

    for (int l = isused.size() - 1; l >= 0; l--) {
      if (!isused[l]) {
        for (auto i : groups) {
          // cluster allocations
          for (int j = 0; j < n_by_group[i]; j++) {
            if (s[i][j] >= l) {
              s[i][j] -= 1;
            }
          }
          // theta_star
          theta_star[i].erase(theta_star[i].begin() + l);

          // maybe theta tilde
          if (v[i][l] >= 0) {
            // std::cout << "decreasing v by one if greater than: " <<
            // v[i][l]
            // << std::endl;
            for (int k = 0; k < v[i].size(); k++) {
              if (v[i][k] > v[i][l]) {
                v[i][k] -= 1;
              }
            }
            // std::cout << "removing theta_tilde " << i << ", " << v[i][l]
            //           << std::endl;
            theta_tilde[i].erase(theta_tilde[i].begin() + v[i][l]);
          }
          // v variables
          v[i].erase(v[i].begin() + l);

          // t variables
          t[i].erase(t[i].begin() + l);
        }
      }
    }
  }

  // find the taus which are not allocated and delete them
  std::vector<bool> isused_tau(taus.size(), false);
  for (int i = 0; i < ngroups; i++) {
    for (int l = 0; l < t[i].size(); l++) {
      if (t[i][l] >= 0) {
        isused_tau[t[i][l]] = true;
      }
    }
  }
  //   std::cout << "Unused taus: ";
  //   for (int l = 0; l < isused_tau.size(); l++) {
  //     if (!isused_tau[l]) std::cout << l << ", ";
  //   }
  //   std::cout << std::endl;

  for (int l = isused_tau.size() - 1; l >= 0; l--) {
    if (!isused_tau[l]) {
      taus.erase(taus.begin() + l);
      for (int i = 0; i < t.size(); i++) {
        for (int k = 0; k < t[i].size(); k++) {
          if (t[i][k] >= l) {
            t[i][k] -= 1;
          }
        }
      }
    }
  }

  //   std::cout << "t: " << std::endl;
  //   for (int i = 0; i < ngroups; i++) {
  //     for (auto k : t[i]) std::cout << k << ", ";
  //     std::cout << std::endl;
  //   }

  //   for (int i = 0; i < ngroups; i++) {
  //     std::cout << "group: " << i << ",theta_stars: " <<
  //     theta_star[i].size()
  //               << ", theta_tilde: " << theta_tilde[i].size() <<
  //               std::endl;
  //   }

  //   std::cout << "taus.size(): " << taus.size() << std::endl;
}