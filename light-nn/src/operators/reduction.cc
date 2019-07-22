#include "operators/reduction.h"
#include "utils/math-functions.h"

namespace lnn {

Reduction::Reduction(const Json::Value &config) : Operator(config) {
  m_name = config["name"].asString();
  if (config["param"].isMember("type")) {
    m_type = config["param"]["type"].asString();
  } else {
    m_type = "LAST";
  }
}

Reduction::~Reduction() {
}

bool Reduction::set_weight(const std::vector<Tensor> &weights,
                           const std::map<std::string, size_t> &weights_name2id) {
  return true;
}

bool Reduction::reshape(const std::vector<Tensor *> &input,
                        std::vector<Tensor *> &output) {
  if (2 != input[0]->num_axes()) {
    LOG(ERROR) << "Only support 2d input of shape T*D!" << std::endl;
    return false;
  }
  std::vector<size_t> shape(1, input[0]->shape(1));
  output[0]->realloc(shape);
  return true;
}

void Reduction::forward_impl(const std::vector<Tensor *> &input,
                             std::vector<Tensor *> &output) {
  size_t T = input[0]->shape(0);
  size_t D = input[0]->shape(1);
  if ("LAST" == m_type) {
    lnn_copy(D, input[0]->data() + (T-1)*D, output[0]->data());
  } else {
    lnn_copy(D, input[0]->data(), output[0]->data());
    for (size_t i = 1; i < T; ++i) {
      lnn_add(D, input[0]->data() + i*D, output[0]->data(), output[0]->data());
    }
    if ("MEAN" == m_type) {
      lnn_scal(D, 1.0/T, output[0]->data());
    }
  }
#ifdef DEBUG
  dump(output, std::cout);
#endif  // DEBUG
}

}  // namespace lnn
