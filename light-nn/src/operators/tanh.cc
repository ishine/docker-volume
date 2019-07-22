#include "operators/tanh.h"
#include "utils/math-functions.h"

namespace lnn {

Tanh::Tanh(const Json::Value &config) : Operator(config) {
  m_name = config["name"].asString();
}

Tanh::~Tanh() {
}

bool Tanh::set_weight(const std::vector<Tensor> &weights,
                      const std::map<std::string, size_t> &weights_name2id) {
  // there are no weights for activation function
  return true;
}

bool Tanh::reshape(const std::vector<Tensor *> &input,
                   std::vector<Tensor *> &output) {
  output[0]->realloc(input[0]->shape());
  return true;
}

void Tanh::forward_impl(const std::vector<Tensor *> &input,
                        std::vector<Tensor *> &output) {
  lnn_tanh(output[0]->size(), input[0]->data(), output[0]->data());
#ifdef DEBUG
  dump(output, std::cout);
#endif  // DEBUG
}

}  // namespace lnn
