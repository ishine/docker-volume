// Copyright (c) 2017, Tencent Inc.
// All Rights Reserved
//
// Author: Wenfeng Xuan <johnxuan@tencent.com>
//
#ifndef LNN_TENSOR_H_
#define LNN_TENSOR_H_

#include <stddef.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace lnn {

class Tensor {
 public:
  Tensor() : m_name(""), m_data(NULL), m_num_element(0), b_external(true) {}
  Tensor(const std::string &name) : m_name(name), m_data(NULL),
                                    m_num_element(0), b_external(true) {}
  ~Tensor();
  Tensor& operator=(const Tensor& other);

  inline const std::string & name() { return m_name; }
  inline size_t size() { return m_num_element; }
  inline float* data() { return m_data; }
  inline size_t num_axes() { return m_shape.size(); }
  inline const std::vector<size_t> & shape() { return m_shape; }

  void set_name(const std::string &name) { m_name = name; }
  void set_shape(const std::vector<size_t> &shape) { m_shape = shape; }
  void set_data(float *data, size_t num_element);
  void set_data(float *data, const std::vector<size_t> &shape);
  void get_data(float *data);

  size_t shape(int axis);
  size_t count(size_t begin_axis, size_t end_axis);
  size_t count(size_t axis);
  bool reshape(const std::vector<size_t> &shape);
  void realloc(const std::vector<size_t> &shape);
  size_t canonical_axis(int axis);

  void dump(std::ostream &ofs, bool is_float = true);

 private:
  std::string m_name;
  float* m_data;
  size_t m_num_element;
  std::vector<size_t> m_shape;
  bool b_external;
};

}  // namespace lnn

#endif  // LNN_TENSOR_H_
