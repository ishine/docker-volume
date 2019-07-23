// Copyright (c) 2017, Tencent Inc.
// All Rights Reserved
//
// Author: Wenfeng Xuan <johnxuan@tencent.com>
//
#ifndef LNN_OPERATORS_CONV1D_H_
#define LNN_OPERATORS_CONV1D_H_

#include "operator.h"

namespace lnn {
//operator is class
class Conv1D : public Operator {
 public:
  Conv1D(const Json::Value &config);
  virtual ~Conv1D();

  virtual inline const char * type() const { return "Conv1D"; }

  virtual bool set_weight(const std::vector<Tensor> &weights,
                          const std::map<std::string, size_t> &weights_name2id);

  virtual bool reshape(const std::vector<Tensor *> &input,
                       std::vector<Tensor *> &output);

 private:
  virtual void forward_impl(const std::vector<Tensor *> &input,
                            std::vector<Tensor *> &output);

  bool b_bias;
  size_t m_kernel_size;
  size_t m_stride;
  size_t m_padding;
  size_t m_dilation;
  Tensor m_bias_multiplier;
  Tensor m_buf, m_buf_2;
};

}  // namespace lnn

#endif  // LNN_OPERATORS_CONV1D_H_
