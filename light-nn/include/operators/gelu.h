// Copyright (c) 2017, Tencent Inc.
// All Rights Reserved
//
// Author: Wenfeng Xuan <johnxuan@tencent.com>
//
#ifndef LNN_OPERATORS_GELU_H_
#define LNN_OPERATORS_GELU_H_

#include "operator.h"

namespace lnn {

// Implement the activation function: GELU(x) = x * P(X <= x),
// where P() is cumulative distribution function, and X obeys
// Gaussian distribution.
//
// Reference: Bridging nonlinearities and stochastic regularizers
// with Gaussian Error Linear Units.
//
// Links: https://arxiv.org/abs/1606.08415
class GELU : public Operator {
 public:
  GELU(const Json::Value &config);
  virtual ~GELU();

  virtual inline const char * type() const { return "GELU"; }

  virtual bool set_weight(const std::vector<Tensor> &weights,
                          const std::map<std::string, size_t> &weights_name2id);

  virtual bool reshape(const std::vector<Tensor *> &input,
                       std::vector<Tensor *> &output);

 private:
  virtual void forward_impl(const std::vector<Tensor *> &input,
                            std::vector<Tensor *> &output);

  float m_mean;
  float m_deviation;
};

}  // namespace lnn

#endif  // LNN_OPERATORS_GELU_H_
