#include "operators/concat.h"
#include "utils/math-functions.h"

namespace lnn {

Concat::Concat(const Json::Value &config) : Operator(config) {
  m_name = config["name"].asString();
  if (config["param"].isMember("axis")) {
    m_axis = config["param"]["axis"].asInt();
  } else {
    m_axis = -1;
  }
}

Concat::~Concat() {
}

bool Concat::set_weight(const std::vector<Tensor> &weights,
                        const std::map<std::string, size_t> &weights_name2id) {
  return true;
}

bool Concat::reshape(const std::vector<Tensor *> &input,
                     std::vector<Tensor *> &output) {
  size_t num_axes = input[0]->num_axes();
  for (size_t i = 1; i < input.size(); ++i) {
    if (num_axes != input[i]->num_axes()) {
      LOG(ERROR) << "All inputs must have the same #axes!" << std::endl;
      return false;
    }
  }
  m_axis = input[0]->canonical_axis(m_axis);
  m_num_concats = input[0]->count(0, m_axis);
  m_concat_size = input[0]->count(m_axis + 1);
  std::vector<size_t> output_shape = input[0]->shape();
  for (size_t i = 1; i < input.size(); ++i) {
    for (size_t j = 0; j < num_axes; ++j) {
      if (static_cast<size_t>(m_axis) == j) {
        output_shape[m_axis] += input[i]->shape(j);
        continue;
      }
      if (output_shape[j] != input[i]->shape(j)) {
        LOG(ERROR) << "All inputs must have the same shape, except at m_axis!" << std::endl;
        return false;
      }
    }
  }
  output[0]->realloc(output_shape);
  return true;
}

void Concat::forward_impl(const std::vector<Tensor *> &input,
                          std::vector<Tensor *> &output) {
  size_t offset_concat_axis = 0;
  size_t output_concat_axis = output[0]->shape(m_axis);
  for (size_t i = 0; i < input.size(); ++i) {
    size_t input_concat_axis = input[i]->shape(m_axis);
    for (size_t n = 0; n < m_num_concats; ++n) {
      lnn_copy(input_concat_axis * m_concat_size,
               input[i]->data() + n * input_concat_axis * m_concat_size,
               output[0]->data() + (n * output_concat_axis + offset_concat_axis) * m_concat_size);
    }
    offset_concat_axis += input_concat_axis;
  }
#ifdef DEBUG
  dump(output, std::cout);
#endif  // DEBUG
}

}  // namespace lnn
