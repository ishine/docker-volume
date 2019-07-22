#include "operators/lookup-table.h"
#include "utils/math-functions.h"

namespace lnn {

LookupTable::LookupTable(const Json::Value &config) : Operator(config) {
  m_name = config["name"].asString();
}

LookupTable::~LookupTable() {
}

bool LookupTable::set_weight(const std::vector<Tensor> &weights,
                             const std::map<std::string, size_t> &weights_name2id) {
  m_weights.resize(1);

  // get tensors needed by current operator
  std::map<std::string, size_t>::const_iterator it;
  get_tensor(m_name + ".weight", m_name, "LookupTable", m_weights[0]);
  // set weight tensor's shape
  std::vector<size_t> shape(1, m_weights[0]->size());
  m_weights[0]->set_shape(shape);
  return true;
}

bool LookupTable::reshape(const std::vector<Tensor *> &input,
                          std::vector<Tensor *> &output) {
  for (size_t i = 0; i < input[0]->size(); ++i) {
    int val = input[0]->data()[i];
    if (val < 0 || static_cast<size_t>(val) >= m_weights[0]->size()) {
      LOG(ERROR) << "Invalid index [" << val << "], it should be in [0, "
        << m_weights[0]->size() << ")!" << std::endl;
      return false;
    }
  }
  std::vector<size_t> shape(input[0]->shape());
  // from L*C*T to L*C
  shape.resize(2);
  output[0]->realloc(shape);
  return true;
}

void LookupTable::forward_impl(const std::vector<Tensor *> &input,
                               std::vector<Tensor *> &output) {
  size_t L = input[0]->shape(0);
  size_t C = input[0]->shape(1);
  size_t T = input[0]->shape(2);
  lnn_set(output[0]->size(), 0., output[0]->data());
  for (size_t i = 0; i < L; ++i) {
    for (size_t j = 0; j < C; ++j) {
      for (size_t k = 0; k < T; ++k) {
        int idx = input[0]->data()[i*C*T+j*T+k];
        output[0]->data()[i*C+j] += m_weights[0]->data()[idx];
      }
    }
  }
#ifdef DEBUG
  dump(output, std::cout);
#endif  // DEBUG
}

}  // namespace lnn
