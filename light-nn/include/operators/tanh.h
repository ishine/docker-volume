// Copyright (c) 2017, Tencent Inc.
// All Rights Reserved
//
// Author: Wenfeng Xuan <johnxuan@tencent.com>
//
#ifndef LNN_OPERATORS_TANH_H_
#define LNN_OPERATORS_TANH_H_

#include "operator.h"

namespace lnn {

class Tanh : public Operator {
 public:
  Tanh(const Json::Value &config);
  virtual ~Tanh();

  virtual inline const char * type() const { return "Tanh"; }

  virtual bool set_weight(const std::vector<Tensor> &weights,
                          const std::map<std::string, size_t> &weights_name2id);

  virtual bool reshape(const std::vector<Tensor *> &input,
                       std::vector<Tensor *> &output);

 private:
  virtual void forward_impl(const std::vector<Tensor *> &input,
                            std::vector<Tensor *> &output);
};

}  // namespace lnn

#endif  // LNN_OPERATORS_TANH_H_
