// Copyright (c) 2017, Tencent Inc.
// All Rights Reserved
//
// Author: Wenfeng Xuan <johnxuan@tencent.com>
//
#ifndef LNN_OPERATORS_CRF_H_
#define LNN_OPERATORS_CRF_H_

#include "operator.h"

namespace lnn {

class CRF : public Operator {
 public:
  CRF(const Json::Value &config);
  virtual ~CRF();

  virtual inline const char * type() const { return "CRF"; }

  virtual bool set_weight(const std::vector<Tensor> &weights,
                          const std::map<std::string, size_t> &weights_name2id);

  virtual bool reshape(const std::vector<Tensor *> &input,
                       std::vector<Tensor *> &output);

 private:
  virtual void forward_impl(const std::vector<Tensor *> &input,
                            std::vector<Tensor *> &output);

  size_t m_label_size;
  Tensor m_opt_val, m_opt_idx;
};

}  // namespace lnn

#endif  // LNN_OPERATORS_CRF_H_
