// Copyright (c) 2017, Tencent Inc.
// All Rights Reserved
//
// Author: Wenfeng Xuan <johnxuan@tencent.com>
//
#ifndef LNN_OPERATORS_WEIGHT_WRAPPER_H_
#define LNN_OPERATORS_WEIGHT_WRAPPER_H_

#include "operator.h"

namespace lnn {

class WeightWrapper : public Operator {
 public:
  WeightWrapper(const Json::Value &config);
  virtual ~WeightWrapper();

  virtual inline const char * type() const { return "WeightWrapper"; }

  virtual bool set_weight(const std::vector<Tensor> &weights,
                          const std::map<std::string, size_t> &weights_name2id);

  virtual bool reshape(const std::vector<Tensor *> &input,
                       std::vector<Tensor *> &output);

 private:
  virtual void forward_impl(const std::vector<Tensor *> &input,
                            std::vector<Tensor *> &output);

  std::string m_weight_name;
};

}  // namespace lnn

#endif  // LNN_OPERATORS_WEIGHT_WRAPPER_H_
