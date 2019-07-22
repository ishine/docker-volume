#include "utils/math-functions.h"

#include <float.h>
#include <math.h>
#include <string.h>

#include "config.h"
#include "cblas.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif  // USE_OPENMP

namespace lnn {

void lnn_set(const int n, const float alpha, float* x) {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif  // USE_OPENMP
  for (int i = 0; i < n; ++i) {
    x[i] = alpha;
  }
}

void lnn_memset(const size_t n, const int alpha, void* x) {
  memset(x, alpha, n);
}

void lnn_copy(const int n, const float* x, float* y) {
  cblas_scopy(n, x, 1, y, 1);
}

void lnn_gemm(const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
              const int m, const int n, const int k, const float alpha,
              const float* a, const float* b, const float beta, float* c) {
  int lda = (transa == CblasNoTrans) ? k : m;
  int ldb = (transb == CblasNoTrans) ? n : k;
  cblas_sgemm(CblasRowMajor, transa, transb, m, n, k, alpha, a, lda, b,
              ldb, beta, c, n);
}

void lnn_gemm(const int m, const int n, const int k, const float alpha,
              const float* a, const float* b, const float beta, float* c) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, k, b,
              n, beta, c, n);
}

void lnn_gemv(const CBLAS_TRANSPOSE trans, const int m, const int n,
              const float alpha, const float* a, const float* x,
              const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, trans, m, n, alpha, a, n, x, 1, beta, y, 1);
}

void lnn_gemv(const int m, const int n, const float alpha, const float* a,
              const float* x, const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, a, n, x, 1, beta, y, 1);
}

void lnn_axpy(const int n, const float alpha, const float* x, float* y) {
  cblas_saxpy(n, alpha, x, 1, y, 1);
}

void lnn_axpby(const int n, const float alpha, const float* x,
               const float beta, float* y) {
  cblas_saxpby(n, alpha, x, 1, beta, y, 1);
}

void lnn_add_scalar(const int n, const float alpha, float* x) {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif  // USE_OPENMP
  for (int i = 0; i < n; ++i) {
    x[i] += alpha;
  }
}

void lnn_scal(const int n, const float alpha, float *x) {
  cblas_sscal(n, alpha, x, 1);
}

void lnn_scal(const int n, const float alpha, const float *x, float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

void lnn_add(const int n, const float* a, const float* b, float* y) {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif  // USE_OPENMP
  for (int i = 0; i < n; ++i) {
    y[i] = a[i] + b[i];
  }
}

void lnn_sub(const int n, const float* a, const float* b, float* y) {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif  // USE_OPENMP
  for (int i = 0; i < n; ++i) {
    y[i] = a[i] - b[i];
  }
}

void lnn_mul(const int n, const float* a, const float* b, float* y) {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif  // USE_OPENMP
  for (int i = 0; i < n; ++i) {
    y[i] = a[i] * b[i];
  }
}

void lnn_div(const int n, const float* a, const float* b, float* y) {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif  // USE_OPENMP
  for (int i = 0; i < n; ++i) {
    y[i] = a[i] / b[i];
  }
}

void lnn_powx(const int n, const float* a, const float b, float* y) {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif  // USE_OPENMP
  for (int i = 0; i < n; ++i) {
    y[i] = powf(a[i], b);
  }
}

void lnn_exp(const int n, const float* a, float* y) {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif  // USE_OPENMP
  for (int i = 0; i < n; ++i) {
    y[i] = expf(a[i]);
  }
}

void lnn_log(const int n, const float* a, float* y) {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif  // USE_OPENMP
  for (int i = 0; i < n; ++i) {
    y[i] = logf(a[i]);
  }
}

void lnn_sqr(const int n, const float* a, float* y) {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif  // USE_OPENMP
  for (int i = 0; i < n; ++i) {
    y[i] = powf(a[i], 2.0);
  }
}

void lnn_sqrt(const int n, const float* a, float* y) {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif  // USE_OPENMP
  for (int i = 0; i < n; ++i) {
    y[i] = sqrtf(a[i]);
  }
}

void lnn_abs(const int n, const float* a, float* y) {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif  // USE_OPENMP
  for (int i = 0; i < n; ++i) {
    y[i] = fabs(a[i]);
  }
}

float lnn_strided_dot(const int n, const float* x, const int incx,
                      const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

float lnn_dot(const int n, const float* x, const float* y) {
  return lnn_strided_dot(n, x, 1, y, 1);
}

float lnn_asum(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

float sigmoid(float x) {
  return 1. / (1. + exp(-x));
}

void lnn_sigmoid(const int n, const float* x, float* y) {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif  // USE_OPENMP
  for (int i = 0; i < n; ++i) {
    y[i] = sigmoid(x[i]);
  }
}

void lnn_tanh(const int n, const float* x, float* y) {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif  // USE_OPENMP
  for (int i = 0; i < n; ++i) {
    y[i] = tanh(x[i]);
  }
}

void lnn_relu(const int n, const float* x, float* y) {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif  // USE_OPENMP
  for (int i = 0; i < n; ++i) {
    y[i] = (x[i] > 0. ? x[i] : 0.);
  }
}

void lnn_gelu(const int n, const float* x, float* y,
              const float mean, const float deviation) {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif  // USE_OPENMP
  for (int i = 0; i < n; ++i) {
    y[i] = x[i] * 0.5 * (1. + erf((x[i] - mean) / (1.414214 * deviation)));
  }
}

void lnn_softmax(const int n, const float* x, float* y) {
  lnn_copy(n, x, y);
  float max_val = -FLT_MAX;
  for (int i = 0; i < n; ++i) {
    if (y[i] > max_val) max_val = y[i];
  }
#ifdef USE_OPENMP
#pragma omp parallel for
#endif  // USE_OPENMP
  for (int i = 0; i < n; ++i) {
    y[i] -= max_val;
    y[i] = exp(y[i]);
  }
  float sum = 0.;
  for (int i = 0; i < n; ++i) { sum += y[i]; }
#ifdef USE_OPENMP
#pragma omp parallel for
#endif  // USE_OPENMP
  for (int i = 0; i < n; ++i) { y[i] /= sum; }
}

}  // namespace lnn
