// Copyright (c) 2017, Tencent Inc.
// All Rights Reserved
//
// Author: Wenfeng Xuan <johnxuan@tencent.com>
//
#ifndef LNN_EXECUTOR_H_
#define LNN_EXECUTOR_H_

#include <vector>

#include "tensor.h"

namespace lnn {

class Net;
class Operator;

class Executor {
 public:
  Executor(const Net* net, int num_threads = 1);
  ~Executor();

  const std::vector<Tensor *> & execute(const std::vector<Tensor> &input,
                                        bool &success);

 private:
  Net* m_net;
  int m_num_threads;

  std::vector<Tensor> m_dynamic_tensors;
  std::vector<Operator *> m_operators;

  std::vector<Tensor *> m_output_tensors;
};

}  // namespace lnn

#endif  // LNN_EXECUTOR_H_
