// Copyright (c) 2017, Tencent Inc.
// All Rights Reserved
//
// Author: Wenfeng Xuan <johnxuan@tencent.com>
//
#ifndef LNN_UTILS_MATH_FUNCTIONS_H_
#define LNN_UTILS_MATH_FUNCTIONS_H_

#include <stdint.h>

#include "cblas.h"

namespace lnn {

void lnn_set(const int n, const float alpha, float *x);

void lnn_memset(const size_t n, const int alpha, void* x);

void lnn_copy(const int n, const float *x, float *y);

void lnn_gemm(const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
              const int m, const int n, const int k, const float alpha,
              const float* a, const float* b, const float beta, float* c);

void lnn_gemm(const int m, const int n, const int k, const float alpha,
              const float* a, const float* b, const float beta, float* c);

void lnn_gemv(const CBLAS_TRANSPOSE trans, const int m, const int n,
              const float alpha, const float* a, const float* x,
              const float beta, float* y);

void lnn_gemv(const int m, const int n, const float alpha, const float* a,
              const float* x, const float beta, float* y);

void lnn_axpy(const int n, const float alpha, const float* x, float* y);

void lnn_axpby(const int n, const float alpha, const float* x, const float beta,
               float* y);

void lnn_add_scalar(const int n, const float alpha, float *x);

void lnn_scal(const int n, const float alpha, float *x);

void lnn_scal(const int n, const float alpha, const float *x, float* y);

void lnn_sqr(const int n, const float* a, float* y);

void lnn_sqrt(const int n, const float* a, float* y);

void lnn_add(const int n, const float* a, const float* b, float* y);

void lnn_sub(const int n, const float* a, const float* b, float* y);

void lnn_mul(const int n, const float* a, const float* b, float* y);

void lnn_div(const int n, const float* a, const float* b, float* y);

void lnn_powx(const int n, const float* a, const float b, float* y);

void lnn_exp(const int n, const float* a, float* y);

void lnn_log(const int n, const float* a, float* y);

void lnn_abs(const int n, const float* a, float* y);

float lnn_strided_dot(const int n, const float* x, const int incx,
                      const float* y, const int incy);

float lnn_dot(const int n, const float* x, const float* y);

float lnn_asum(const int n, const float* x);

void lnn_sigmoid(const int n, const float* x, float* y);

void lnn_tanh(const int n, const float* x, float* y);

void lnn_relu(const int n, const float* x, float* y);

void lnn_leakyrelu(const int n, const float* x, float* y);
void lnn_gelu(const int n, const float* x, float* y,
              const float mean, const float deviation);

void lnn_softmax(const int n, const float* x, float* y);

}  // namespace lnn

#endif  // LNN_UTILS_MATH_FUNCTIONS_H_
