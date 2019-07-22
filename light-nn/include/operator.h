// Copyright (c) 2017, Tencent Inc.
// All Rights Reserved
//
// Author: Wenfeng Xuan <johnxuan@tencent.com>
//
#ifndef LNN_OPERATOR_H_
#define LNN_OPERATOR_H_

#include <map>
#include <string>
#include <vector>

#include "config.h"
#include "json/json.h"
#include "tensor.h"
#include "utils/log.h"

namespace lnn {

class Operator {
 public:
  Operator(const Json::Value &config) {}
  virtual ~Operator() {}

  virtual inline const char * type() const { return ""; }

  // invoked only once
  virtual bool set_weight(const std::vector<Tensor> &weights,
                          const std::map<std::string, size_t> &weights_name2id) = 0;

  // invoked for every input
  //   validate input & reshape output
  virtual bool reshape(const std::vector<Tensor *> &input,
                       std::vector<Tensor *> &output) = 0;

  // invoked for every input
  virtual bool forward(const std::vector<Tensor *> &input,
                       std::vector<Tensor *> &output) {
    if (!reshape(input, output)) return false;
    forward_impl(input, output);
    return true;
  }

 protected:
  // only responsible for computation, without input validation
  virtual void forward_impl(const std::vector<Tensor *> &input,
                            std::vector<Tensor *> &output) = 0;

  void dump(const std::vector<Tensor *> &output, std::ostream &ofs,
            bool is_float = true) {
    ofs << m_name << std::endl;
    for (size_t i = 0; i < output.size(); ++i) {
      if (output.size() > 1) {
        ofs << "result of output " << i << std::endl;
      }
      output[i]->dump(ofs, is_float);
    }
  }

  std::string m_name;
  std::vector<Tensor *> m_weights;
  size_t m_input_size;
  size_t m_output_size;
};

#define get_tensor(tensor_name, operator_name, operator_type, dest) \
  it = weights_name2id.find(tensor_name);                           \
  if (weights_name2id.end() == it) {                                \
    LOG(ERROR) << "Missing tensor [" << tensor_name << "] for"      \
      << " operator [" << operator_name << "] of type ["            \
      << operator_type << "]!" << std::endl;                        \
    return false;                                                   \
  }                                                                 \
  dest = const_cast<Tensor *>(&(weights[it->second]))

}  // namespace lnn

#endif  // LNN_OPERATOR_H_
