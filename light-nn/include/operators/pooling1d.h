// Copyright (c) 2017, Tencent Inc.
// All Rights Reserved
//
// Author: Wenfeng Xuan <johnxuan@tencent.com>
//
#ifndef LNN_OPERATORS_POOLING1D_H_
#define LNN_OPERATORS_POOLING1D_H_

#include "operator.h"

namespace lnn {

class Pooling1D : public Operator {
 public:
  Pooling1D(const Json::Value &config);
  virtual ~Pooling1D();

  virtual inline const char * type() const { return "Pooling1D"; }

  virtual bool set_weight(const std::vector<Tensor> &weights,
                          const std::map<std::string, size_t> &weights_name2id);

  virtual bool reshape(const std::vector<Tensor *> &input,
                       std::vector<Tensor *> &output);

 private:
  virtual void forward_impl(const std::vector<Tensor *> &input,
                            std::vector<Tensor *> &output);
  void forward_max(const std::vector<Tensor *> &input,
                   std::vector<Tensor *> &output);
  void forward_mean(const std::vector<Tensor *> &input,
                    std::vector<Tensor *> &output);

  std::string m_type;  // {"MAX", "MEAN"}, default is MAX
  bool b_global_pooling;
  size_t m_kernel_size;
  size_t m_stride;
  size_t m_padding;
  size_t m_dilation;
  Tensor m_buf;
};

}  // namespace lnn

#endif  // LNN_OPERATORS_POOLING1D_H_
