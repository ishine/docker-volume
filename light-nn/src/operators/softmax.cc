#include "operators/softmax.h"
#include "utils/math-functions.h"

namespace lnn {

Softmax::Softmax(const Json::Value &config) : Operator(config) {
  m_name = config["name"].asString();
  if (config["param"].isMember("axis")) {
    m_axis = config["param"]["axis"].asInt();
  } else {
    m_axis = -1;  // means the last axis
  }
}

Softmax::~Softmax() {
}

bool Softmax::set_weight(const std::vector<Tensor> &weights,
                        const std::map<std::string, size_t> &weights_name2id) {
  // there are no weights for activation function
  return true;
}

bool Softmax::reshape(const std::vector<Tensor *> &input,
                      std::vector<Tensor *> &output) {
  output[0]->realloc(input[0]->shape());
  return true;
}

void Softmax::forward_impl(const std::vector<Tensor *> &input,
                           std::vector<Tensor *> &output) {
  m_axis = input[0]->canonical_axis(m_axis);
  size_t cls_cnt = input[0]->count(m_axis);
  size_t batch = input[0]->size() / cls_cnt;
  for (size_t i = 0; i < batch; ++i) {
    lnn_softmax(cls_cnt, input[0]->data() + i * cls_cnt,
                output[0]->data() + i * cls_cnt);
  }
#ifdef DEBUG
  dump(output, std::cout);
#endif  // DEBUG
}

}  // namespace lnn
