// Copyright (c) 2017, Tencent Inc.
// All Rights Reserved
//
// Author: Wenfeng Xuan <johnxuan@tencent.com>
//
#ifndef LNN_OPERATORS_LSTM_H_
#define LNN_OPERATORS_LSTM_H_

#include "operator.h"

namespace lnn {

class LSTM : public Operator {
 public:
  LSTM(const Json::Value &config);
  virtual ~LSTM();

  virtual inline const char * type() const { return "LSTM"; }

  virtual bool set_weight(const std::vector<Tensor> &weights,
                          const std::map<std::string, size_t> &weights_name2id);

  virtual bool reshape(const std::vector<Tensor *> &input,
                       std::vector<Tensor *> &output);

 private:
  virtual void forward_impl(const std::vector<Tensor *> &input,
                            std::vector<Tensor *> &output);

  bool b_bidirectional;
  bool b_bias;
  Tensor m_bias_multiplier;
  Tensor m_h_0, m_c_0;
  Tensor m_buf, m_buf_reverse;
  Tensor m_cell, m_ig;
  bool b_with_cache;
  bool *b_cached, *b_cached_reverse;
  float *m_cache, *m_cache_reverse;
  size_t m_top_k;
};

}  // namespace lnn

#endif  // LNN_OPERATORS_LSTM_H_
