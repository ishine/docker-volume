// Copyright (c) 2017, Tencent Inc.
// All Rights Reserved
//
// Author: Wenfeng Xuan <johnxuan@tencent.com>
//
#ifndef LNN_H_
#define LNN_H_

#include "executor.h"
#include "net.h"
#include "tensor.h"

namespace lnn {

// c = alpha * (a * b) + beta * c
// 'a' of shape m*k, 'b' of shape k*n
void lnn_gemm(const int m, const int n, const int k, const float alpha,
              const float* a, const float* b, const float beta, float* c);
// y = alpha * (a * x) + beta * y
// 'a' of shape m*n, 'x' of shape n*1
void lnn_gemv(const int m, const int n, const float alpha, const float* a,
              const float* x, const float beta, float* y);
void lnn_sigmoid(const int n, const float* x, float* y);
void lnn_tanh(const int n, const float* x, float* y);
void lnn_relu(const int n, const float* x, float* y);
void lnn_softmax(const int n, const float* x, float* y);

float lnn_dot(const int n, const float* x, const float* y);

}  // namespace lnn

#endif  // LNN_H_
