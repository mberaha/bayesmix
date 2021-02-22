#ifndef BAYESMIX_HIERARCHIES_LOAD_HIERARCHIES_H_
#define BAYESMIX_HIERARCHIES_LOAD_HIERARCHIES_H_

#include <functional>
#include <memory>

#include "base_hierarchy.h"
#include "hierarchy_id.pb.h"
#include "lin_reg_uni_hierarchy.h"
#include "nnig_hierarchy.h"
#include "nnw_hierarchy.h"
#include "src/runtime/factory.h"

template <class AbstractProduct>
using Builder = std::function<std::shared_ptr<AbstractProduct>()>;

using HierarchyFactory = Factory<bayesmix::HierarchyId, AbstractHierarchy>;

__attribute__((constructor)) static void load_hierarchies() {
  HierarchyFactory &factory = HierarchyFactory::Instance();
  Builder<AbstractHierarchy> NNIGbuilder = []() {
    return std::make_shared<NNIGHierarchy>();
  };
  Builder<AbstractHierarchy> NNWbuilder = []() {
    return std::make_shared<NNWHierarchy>();
  };
  Builder<AbstractHierarchy> LinRegUnibuilder = []() {
    return std::make_shared<LinRegUniHierarchy>();
  };
  factory.add_builder(NNIGHierarchy().get_id(), NNIGbuilder);
  factory.add_builder(NNWHierarchy().get_id(), NNWbuilder);
  factory.add_builder(LinRegUniHierarchy().get_id(), LinRegUnibuilder);
}

#endif  // BAYESMIX_HIERARCHIES_LOAD_HIERARCHIES_H_
