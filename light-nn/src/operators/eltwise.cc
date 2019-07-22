#include "operators/eltwise.h"
#include "utils/math-functions.h"

#include <float.h>

namespace lnn {

Eltwise::Eltwise(const Json::Value &config) : Operator(config) {
  m_name = config["name"].asString();
  m_type = config["param"]["type"].asString();
}

Eltwise::~Eltwise() {
}

bool Eltwise::set_weight(const std::vector<Tensor> &weights,
                         const std::map<std::string, size_t> &weights_name2id) {
  return true;
}

bool Eltwise::reshape(const std::vector<Tensor *> &input,
                      std::vector<Tensor *> &output) {
  if (input.size() < 2) {
    LOG(ERROR) << "Must have at least 2 input tensors!" << std::endl;
    return false;
  }
  for (size_t i = 1; i < input.size(); ++i) {
    if (input[i]->shape() != input[0]->shape()) {
      LOG(ERROR) << "Input tensors should have the same shape!" << std::endl;
      return false;
    }
  }
  output[0]->realloc(input[0]->shape());
  return true;
}

void Eltwise::forward_impl(const std::vector<Tensor *> &input,
                           std::vector<Tensor *> &output) {
  size_t count = output[0]->size();
  float* out_data = output[0]->data();
  if ("PROD" == m_type) {
    lnn_mul(count, input[0]->data(), input[1]->data(), out_data);
    for (size_t i = 2; i < input.size(); ++i) {
      lnn_mul(count, out_data, input[i]->data(), out_data);
    }
  } else if ("SUM" == m_type) {
    lnn_set(count, 0., out_data);
    for (size_t i = 0; i < input.size(); ++i) {
      lnn_add(count, out_data, input[i]->data(), out_data);
    }
  } else {  // "MAX" == m_type
    lnn_set(count, -FLT_MAX, out_data);
    for (size_t idx = 0; idx < count; ++idx) {
      for (size_t i = 0; i < input.size(); ++i) {
        if (input[i]->data()[idx] > out_data[idx]) {
          out_data[idx] = input[i]->data()[idx];
        }
      }
    }
  }
#ifdef DEBUG
  dump(output, std::cout);
#endif  // DEBUG
}

}  // namespace lnn
