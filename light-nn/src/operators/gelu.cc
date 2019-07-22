#include "operators/gelu.h"
#include "utils/math-functions.h"

namespace lnn {

GELU::GELU(const Json::Value &config) : Operator(config) {
  m_name = config["name"].asString();
  m_mean = 0.0;
  if (config["param"].isMember("mean")) {
    m_mean = config["param"]["mean"].asFloat();
  }
  m_deviation = 1.0;
  if (config["param"].isMember("deviation")) {
    m_deviation = config["param"]["deviation"].asFloat();
  }
}

GELU::~GELU() {
}

bool GELU::set_weight(const std::vector<Tensor> &weights,
                      const std::map<std::string, size_t> &weights_name2id) {
  // there are no weights for activation function
  return true;
}

bool GELU::reshape(const std::vector<Tensor *> &input,
                   std::vector<Tensor *> &output) {
  output[0]->realloc(input[0]->shape());
  return true;
}

void GELU::forward_impl(const std::vector<Tensor *> &input,
                        std::vector<Tensor *> &output) {
  lnn_gelu(output[0]->size(), input[0]->data(), output[0]->data(),
           m_mean, m_deviation);
#ifdef DEBUG
  dump(output, std::cout);
#endif  // DEBUG
}

}  // namespace lnn
