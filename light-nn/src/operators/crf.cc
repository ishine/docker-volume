#include "operators/crf.h"
#include "utils/math-functions.h"

#include <float.h>

namespace lnn {

CRF::CRF(const Json::Value &config) : Operator(config) {
  m_name = config["name"].asString();
  m_label_size = config["param"]["label_size"].asInt();
}

CRF::~CRF() {
}

bool CRF::set_weight(const std::vector<Tensor> &weights,
                     const std::map<std::string, size_t> &weights_name2id) {
  m_weights.resize(1);

  // get tensors needed by current operator
  std::map<std::string, size_t>::const_iterator it;
  get_tensor(m_name + ".weight", m_name, "CRF", m_weights[0]);
  // check consistency of weight tensor's size
  if (m_weights[0]->size() != (m_label_size+1) * m_label_size) {
    LOG(ERROR) << "Size mismatch of tensor [" << m_weights[0]->name()
      << "] between weight file and model file (" << m_weights[0]->size()
      << ", " << (m_label_size+1) * m_label_size << ")!" << std::endl;
    return false;
  }
  // set weight tensor's shape
  std::vector<size_t> shape;
  shape.push_back(m_label_size+1);
  shape.push_back(m_label_size);
  m_weights[0]->set_shape(shape);
  return true;
}

bool CRF::reshape(const std::vector<Tensor *> &input,
                  std::vector<Tensor *> &output) {
  if (2 != input[0]->num_axes()) {
    LOG(ERROR) << "Only support 2D input of shape T*C!" << std::endl;
    return false;
  }
  if (m_label_size != input[0]->shape(1)) {
    LOG(ERROR) << "Label size mismatch between config and compute: ("
      << m_label_size << ", " << input[0]->shape(1) << ")!" << std::endl;
    return false;
  }
  std::vector<size_t> shape(1, input[0]->shape(0));
  output[0]->realloc(shape);
  m_opt_val.realloc(input[0]->shape());
  m_opt_idx.realloc(input[0]->shape());
  return true;
}

void CRF::forward_impl(const std::vector<Tensor *> &input,
                       std::vector<Tensor *> &output) {
  size_t T = input[0]->shape(0);
  const float* emit = input[0]->data();
  const float* trans = m_weights[0]->data();
  const float* init = trans + m_label_size * m_label_size;
  float* labels_id = output[0]->data();
  float* opt_val = m_opt_val.data();
  float* opt_idx = m_opt_idx.data();
  lnn_set(m_opt_val.size(), -FLT_MAX, opt_val);
  lnn_set(m_opt_idx.size(), -1, opt_idx);
  // initialization
#ifdef USE_OPENMP
#pragma omp parallel for
#endif  // USE_OPENMP
  for (long l = 0; l < m_label_size; ++l) {
    opt_val[l] = init[l] + emit[l];
  }
  // recursion
  for (long w = 1; w < T; ++w) {
    for (long l = 0; l < m_label_size; ++l) {
      float max_val = -FLT_MAX;
      int max_idx = -1;
      for (long pre_l = 0; pre_l < m_label_size; ++pre_l) {
        float tmp = opt_val[pre_l + (w-1) * m_label_size] + trans[l + pre_l * m_label_size];
        if (tmp > max_val) {
          max_val = tmp;
          max_idx = pre_l;
        }
      }
      opt_val[l + w * m_label_size] = max_val + emit[l + w * m_label_size];
      opt_idx[l + w * m_label_size] = max_idx;
    }
  }
  // path backtracking
  float val = -FLT_MAX;
  for (long l = 0; l < m_label_size; ++l) {
    if (opt_val[l + (T - 1) * m_label_size] > val) {
      val = opt_val[l + (T - 1) * m_label_size];
      labels_id[T - 1] = l;
    }
  }
  for (int w = T-2; w >= 0; --w) {
    labels_id[w] = opt_idx[int(labels_id[w+1]) + (w+1) * m_label_size];
  }
#ifdef DEBUG
  std::cout << "opt_val" << std::endl;
  m_opt_val.dump(std::cout);
  std::cout << "opt_idx" << std::endl;
  m_opt_idx.dump(std::cout, false);
  dump(output, std::cout);
#endif  // DEBUG
}

}  // namespace lnn
