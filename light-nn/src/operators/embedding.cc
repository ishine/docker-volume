#include "operators/embedding.h"
#include "utils/math-functions.h"

namespace lnn {

Embedding::Embedding(const Json::Value &config) : Operator(config) {
  m_name = config["name"].asString();
  m_input_size = config["param"]["input_size"].asInt();
  m_output_size = config["param"]["output_size"].asInt();
}

Embedding::~Embedding() {
}

bool Embedding::set_weight(const std::vector<Tensor> &weights,
                           const std::map<std::string, size_t> &weights_name2id) {
  m_weights.resize(1);

  // get tensors needed by current operator
  std::map<std::string, size_t>::const_iterator it;
  get_tensor(m_name + ".weight", m_name, "Embedding", m_weights[0]);
  // check consistency of weight tensor's size
  if (m_weights[0]->size() != m_output_size * m_input_size) {
    LOG(ERROR) << "Size mismatch of tensor [" << m_weights[0]->name()
      << "] between weight file and model file (" << m_weights[0]->size()
      << ", " << m_output_size * m_input_size << ")!" << std::endl;
    return false;
  }
  // set weight tensor's shape
  std::vector<size_t> shape;
  shape.push_back(m_input_size);
  shape.push_back(m_output_size);
  m_weights[0]->set_shape(shape);
  return true;
}

bool Embedding::reshape(const std::vector<Tensor *> &input,
                        std::vector<Tensor *> &output) {
  for (size_t i = 0; i < input[0]->size(); ++i) {
    int val = input[0]->data()[i];
    if (val < 0 || static_cast<size_t>(val) >= m_input_size) {
      LOG(ERROR) << "Invalid index [" << val << "], it should be in [0, "
        << m_input_size << ")!" << std::endl;
      return false;
    }
  }
  std::vector<size_t> shape(input[0]->shape());
  shape.push_back(m_output_size);
  output[0]->realloc(shape);
  return true;
}

void Embedding::forward_impl(const std::vector<Tensor *> &input,
                             std::vector<Tensor *> &output) {
  for (size_t i = 0; i < input[0]->size(); ++i) {
    int idx = input[0]->data()[i];
    lnn_copy(m_output_size, m_weights[0]->data() + idx * m_output_size,
             output[0]->data() + i * m_output_size);
  }
#ifdef DEBUG
  dump(output, std::cout);
#endif  // DEBUG
}

}  // namespace lnn
