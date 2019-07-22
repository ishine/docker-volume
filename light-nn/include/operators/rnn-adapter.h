// Copyright (c) 2017, Tencent Inc.
// All Rights Reserved
//
// Author: Wenfeng Xuan <johnxuan@tencent.com>
//
#ifndef LNN_OPERATORS_RNN_ADAPTER_H_
#define LNN_OPERATORS_RNN_ADAPTER_H_

#include "operator.h"

namespace lnn {

class RNNAdapter : public Operator {
 public:
  RNNAdapter(const Json::Value &config);
  virtual ~RNNAdapter();

  virtual inline const char * type() const { return "RNNAdapter"; }

  virtual bool set_weight(const std::vector<Tensor> &weights,
                          const std::map<std::string, size_t> &weights_name2id);

  virtual bool reshape(const std::vector<Tensor *> &input,
                       std::vector<Tensor *> &output);

 private:
  virtual void forward_impl(const std::vector<Tensor *> &input,
                            std::vector<Tensor *> &output);

  std::string m_type;  // {"LAST", "AVERAGE"}, default is the first
};

}  // namespace lnn

#endif  // LNN_OPERATORS_RNN_ADAPTER_H_