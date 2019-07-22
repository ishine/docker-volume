#include "operators/scale.h"
#include "utils/math-functions.h"

namespace lnn {

Scale::Scale(const Json::Value &config) : Operator(config) {
  m_name = config["name"].asString();
  if (config["param"].isMember("scale")) {
    m_scale = config["param"]["scale"].asFloat();
    b_set = true;
  } else {
    m_scale = 0.0;
    b_set = false;
  }
}

Scale::~Scale() {
}

bool Scale::set_weight(const std::vector<Tensor> &weights,
                       const std::map<std::string, size_t> &weights_name2id) {
  return true;
}

bool Scale::reshape(const std::vector<Tensor *> &input,
                    std::vector<Tensor *> &output) {
  if (1 != input.size() && 2 != input.size()) {
    LOG(ERROR) << "Input tensor number can only be 1 or 2!" << std::endl;
    return false;
  }
  if (1 == input.size() && !b_set) {
    LOG(ERROR) << "Input tensor number is 1, you should specify the scale value"
      " in net json file!" << std::endl;
    return false;
  }
  if (2 == input.size()) {
//    LOG(WARNING) << "Input tensor number is 2, the scale value specified in net"
//      " json file will be ignored!" << std::endl;
    if (input[0]->shape(0) != input[1]->shape(0)) {
      LOG(ERROR) << "The first dimension of 2 input tensors should be equal!" << std::endl;
      return false;
    }
  }
  output[0]->realloc(input[0]->shape());
  return true;
}

void Scale::forward_impl(const std::vector<Tensor *> &input,
                         std::vector<Tensor *> &output) {
  if (1 == input.size()) {
    lnn_scal(input[0]->size(), m_scale, input[0]->data(), output[0]->data());
  } else {
    size_t batch = input[0]->shape(0);
    size_t cnt = input[0]->size() / batch;
    for (size_t i = 0; i < batch; ++i) {
      lnn_scal(cnt, input[1]->data()[i], input[0]->data() + i*cnt,
               output[0]->data() + i*cnt);
    }
  }
#ifdef DEBUG
  dump(output, std::cout);
#endif  // DEBUG
}

}  // namespace lnn
