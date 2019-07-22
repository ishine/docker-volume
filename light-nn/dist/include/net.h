// Copyright (c) 2017, Tencent Inc.
// All Rights Reserved
//
// Author: Wenfeng Xuan <johnxuan@tencent.com>
//
#ifndef LNN_NET_H_
#define LNN_NET_H_

#include <map>
#include <string>
#include <vector>

#include "tensor.h"

namespace Json {

class Value;

}  // namespace Json

namespace lnn {

class Net {
 public:
  Net() : m_weight_data(NULL) {}
  ~Net();

  bool load(const char *model_file, const char *weight_file, const char *dir = NULL);

  inline size_t weight_tensor_number()                         { return m_weight_tensors.size(); }
  inline const std::vector<Tensor> & weight_tensors()          { return m_weight_tensors; }
  inline const std::vector<std::string> & weight_tensor_name() { return m_weight_tensor_name; }
  inline const std::map<std::string, size_t> & weight_tensor_name2id() { return m_weight_tensor_name2id; }

  inline size_t dynamic_tensor_number()                            { return m_dynamic_tensor_name.size(); }
  inline const std::vector<std::string> & dynamic_tensor_name()    { return m_dynamic_tensor_name; }
  inline const std::map<std::string, size_t> & dynamic_tensor_name2id() { return m_dynamic_tensor_name2id; }

  inline size_t operator_number()                                  { return m_operator_name.size(); }
  inline const std::vector<std::string> & operator_name()          { return m_operator_name; }

  inline const std::vector<std::vector<size_t> > & op_input_ids()  { return m_input_ids; }
  inline const std::vector<std::vector<size_t> > & op_output_ids() { return m_output_ids; }

  inline const Json::Value * json()                                { return m_json; }

 private:
  bool parse_operators_dependency();

  float *m_weight_data;
  std::vector<Tensor> m_weight_tensors;
  std::vector<std::string> m_weight_tensor_name;
  std::map<std::string, size_t> m_weight_tensor_name2id;

  // size: number of dynamic tensors
  std::vector<std::string> m_dynamic_tensor_name;
  std::map<std::string, size_t> m_dynamic_tensor_name2id;

  // size: number of operators
  std::vector<std::string> m_operator_name;
  // size: number of operators
  std::vector<std::vector<size_t> > m_input_ids;
  // size: number of operators
  std::vector<std::vector<size_t> > m_output_ids;

  Json::Value* m_json;
};

}  // namespace lnn

#endif  // LNN_NET_H_
