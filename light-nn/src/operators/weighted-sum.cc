#include "operators/weighted-sum.h"
#include "utils/math-functions.h"

namespace lnn {

WeightedSum::WeightedSum(const Json::Value &config) : Operator(config) {
  m_name = config["name"].asString();
}

WeightedSum::~WeightedSum() {
}

bool WeightedSum::set_weight(const std::vector<Tensor> &weights,
                             const std::map<std::string, size_t> &weights_name2id) {
  return true;
}

bool WeightedSum::reshape(const std::vector<Tensor *> &input,
                          std::vector<Tensor *> &output) {
  if (3 != input.size()) {
    LOG(ERROR) << "Must have 3 input tensors: tensor1, tensor2, weight_tensor!" << std::endl;
    return false;
  }
  if (input[0]->shape() != input[1]->shape()) {
    LOG(ERROR) << "The first two tensors mush have the same shape!" << std::endl;
    return false;
  }
  if (input[2]->shape() != input[0]->shape()) {
    if (input[2]->size() != 1) {
      LOG(ERROR) << "The weight tensor can be eithor a scalar or have the same shape"
        " with the first two tensors!" << std::endl;
      return false;
    }
  }
  output[0]->realloc(input[0]->shape());
  m_part1.realloc(input[0]->shape());
  m_part2.realloc(input[0]->shape());
  lnn_set(m_part1.size(), 0., m_part1.data());
  lnn_set(m_part2.size(), 1., m_part2.data());
  return true;
}

void WeightedSum::forward_impl(const std::vector<Tensor *> &input,
                               std::vector<Tensor *> &output) {
  if (input[2]->size() == 1) {
    float lambda = input[2]->data()[0];
    lnn_scal(m_part1.size(), lambda, input[0]->data(), m_part1.data());
    lnn_scal(m_part2.size(), 1. - lambda, input[1]->data(), m_part2.data());
    lnn_add(output[0]->size(), m_part1.data(), m_part2.data(), output[0]->data());
  } else {
    lnn_mul(m_part1.size(), input[2]->data(), input[0]->data(), m_part1.data());
    lnn_sub(m_part2.size(), m_part2.data(), input[2]->data(), m_part2.data());
    lnn_mul(m_part2.size(), m_part2.data(), input[1]->data(), m_part2.data());
    lnn_add(output[0]->size(), m_part1.data(), m_part2.data(), output[0]->data());
  }
#ifdef DEBUG
  dump(output, std::cout);
#endif  // DEBUG
}

}  // namespace lnn
