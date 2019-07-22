#include "operators/sigmoid.h"
#include "utils/math-functions.h"

namespace lnn {

Sigmoid::Sigmoid(const Json::Value &config) : Operator(config) {
  m_name = config["name"].asString();
}

Sigmoid::~Sigmoid() {
}

bool Sigmoid::set_weight(const std::vector<Tensor> &weights,
                        const std::map<std::string, size_t> &weights_name2id) {
  // there are no weights for activation function
  return true;
}

bool Sigmoid::reshape(const std::vector<Tensor *> &input,
                      std::vector<Tensor *> &output) {
  output[0]->realloc(input[0]->shape());
  return true;
}

void Sigmoid::forward_impl(const std::vector<Tensor *> &input,
                           std::vector<Tensor *> &output) {
  lnn_sigmoid(output[0]->size(), input[0]->data(), output[0]->data());
#ifdef DEBUG
  dump(output, std::cout);
#endif  // DEBUG
}

}  // namespace lnn
