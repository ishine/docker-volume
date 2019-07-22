#include "operators/reverse.h"
#include "utils/math-functions.h"

namespace lnn {

Reverse::Reverse(const Json::Value &config) : Operator(config) {
  m_name = config["name"].asString();
}

Reverse::~Reverse() {
}

bool Reverse::set_weight(const std::vector<Tensor> &weights,
                         const std::map<std::string, size_t> &weights_name2id) {
  return true;
}

bool Reverse::reshape(const std::vector<Tensor *> &input,
                      std::vector<Tensor *> &output) {
  if (2 != input[0]->num_axes()) {
    LOG(ERROR) << "Only support 2d input of shape T*D!" << std::endl;
    return false;
  }
  output[0]->realloc(input[0]->shape());
  return true;
}

void Reverse::forward_impl(const std::vector<Tensor *> &input,
                           std::vector<Tensor *> &output) {
  size_t T = input[0]->shape(0);
  size_t D = input[0]->shape(1);
  for (size_t i = 0; i < T; ++i) {
    lnn_copy(D, input[0]->data() + i*D, output[0]->data() + (T-1-i)*D);
  }
#ifdef DEBUG
  dump(output, std::cout);
#endif  // DEBUG
}

}  // namespace lnn
