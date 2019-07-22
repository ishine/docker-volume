#include "operators/weight-wrapper.h"
#include "utils/math-functions.h"

namespace lnn {

WeightWrapper::WeightWrapper(const Json::Value &config) : Operator(config) {
  m_name = config["name"].asString();
  m_weight_name = config["param"]["weight_name"].asString();
  m_input_size = config["param"]["input_size"].asInt();
  m_output_size = config["param"]["output_size"].asInt();
  LOG(INFO) << "Only support 1D & 2D weight tensor wrapper!" << std::endl;
}

WeightWrapper::~WeightWrapper() {
}

bool WeightWrapper::set_weight(const std::vector<Tensor> &weights,
                               const std::map<std::string, size_t> &weights_name2id) {
  m_weights.resize(1);
  std::map<std::string, size_t>::const_iterator it;
  get_tensor(m_weight_name, m_name, "WeightWrapper", m_weights[0]);
  std::vector<size_t> shape;
  if (m_output_size > 1) shape.push_back(m_output_size);
  if (m_input_size > 1) shape.push_back(m_input_size);
  m_weights[0]->set_shape(shape);
  return true;
}

bool WeightWrapper::reshape(const std::vector<Tensor *> &input,
                            std::vector<Tensor *> &output) {
  output[0]->realloc(m_weights[0]->shape());
  return true;
}

void WeightWrapper::forward_impl(const std::vector<Tensor *> &input,
                                 std::vector<Tensor *> &output) {
  lnn_copy(m_weights[0]->size(), m_weights[0]->data(), output[0]->data());
#ifdef DEBUG
  dump(output, std::cout);
#endif  // DEBUG
}

}  // namespace lnn
