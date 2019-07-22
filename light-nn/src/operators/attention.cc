#include "operators/attention.h"
#include "utils/math-functions.h"

namespace lnn {

Attention::Attention(const Json::Value &config) : Operator(config) {
  m_name = config["name"].asString();
}

Attention::~Attention() {
}

bool Attention::set_weight(const std::vector<Tensor> &weights,
                           const std::map<std::string, size_t> &weights_name2id) {
  return true;
}

bool Attention::reshape(const std::vector<Tensor *> &input,
                        std::vector<Tensor *> &output) {
  if (2 != input.size()) {
    LOG(ERROR) << "There should be 2 input tensor for attention operator!" << std::endl;
    return false;
  }
  if (2 != input[0]->num_axes() || 1 != input[1]->num_axes()) {
    LOG(ERROR) << "The first input tensor should be with shape T*D, and the second one"
      " should be with shape D!" << std::endl;
    return false;
  }
  if (input[0]->shape(1) != input[1]->shape(0)) {
    LOG(ERROR) << "Dimension mismatch between the two tensor: (" << input[0]->shape(1)
      << ", " << input[1]->shape(0) << ")!" << std::endl;
    return false;
  }
  output[0]->realloc(input[1]->shape());
  std::vector<size_t> shape(1, input[0]->shape(0));
  m_coefficient.realloc(shape);
  return true;
}

void Attention::forward_impl(const std::vector<Tensor *> &input,
                             std::vector<Tensor *> &output) {
  size_t T = input[0]->shape(0);
  size_t D = input[0]->shape(1);
  lnn_gemv(CblasNoTrans, T, D, 1., input[0]->data(), input[1]->data(),
           0., m_coefficient.data());
  lnn_softmax(T, m_coefficient.data(), m_coefficient.data());
  lnn_gemv(CblasTrans, T, D, 1., input[0]->data(), m_coefficient.data(),
           0., output[0]->data());
#ifdef DEBUG
  dump(output, std::cout);
#endif  // DEBUG
}

}  // namespace lnn
