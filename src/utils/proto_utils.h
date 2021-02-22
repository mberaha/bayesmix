#ifndef BAYESMIX_UTILS_PROTO_UTILS_H_
#define BAYESMIX_UTILS_PROTO_UTILS_H_

#include <Eigen/Dense>

#include "matrix.pb.h"

namespace bayesmix {
void to_proto(const Eigen::MatrixXd &mat, bayesmix::Matrix *out);
void to_proto(const Eigen::VectorXd &vec, bayesmix::Vector *out);

Eigen::VectorXd to_eigen(const bayesmix::Vector &vec);
Eigen::MatrixXd to_eigen(const bayesmix::Matrix &mat);

void read_proto_from_file(const std::string &filename,
                          google::protobuf::Message *out);

}  // namespace bayesmix

#endif  // BAYESMIX_UTILS_PROTO_UTILS_H_
