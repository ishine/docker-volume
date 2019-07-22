#include "operators/conv1d.h"
#include "utils/math-functions.h"

namespace lnn {

Conv1D::Conv1D(const Json::Value &config) : Operator(config) {
  m_name = config["name"].asString();
  m_input_size = config["param"]["input_size"].asInt();
  m_output_size = config["param"]["output_size"].asInt();
  m_kernel_size = config["param"]["kernel_size"].asInt();
  if (config["param"].isMember("stride")) {
    m_stride = config["param"]["stride"].asInt();
  } else {
    m_stride = 1;
  }
  if (config["param"].isMember("padding")) {
    m_padding = config["param"]["padding"].asInt();
  } else {
    m_padding = 0;
  }
  if (config["param"].isMember("dilation")) {
    m_dilation = config["param"]["dilation"].asInt();
  } else {
    m_dilation = 1;
  }
  if (config["param"].isMember("bias")) {
    b_bias = config["param"]["bias"].asBool();
  } else {
    b_bias = true;
  }
}

Conv1D::~Conv1D() {
}

bool Conv1D::set_weight(const std::vector<Tensor> &weights,
                        const std::map<std::string, size_t> &weights_name2id) {
  std::vector<std::string> tensor_name;
  std::vector<size_t> tensor_size, shape;
  std::vector<std::vector<size_t> > tensor_shape;
  m_weights.resize(1);
  tensor_name.push_back(m_name + ".weight");
  tensor_size.push_back(m_output_size * m_kernel_size * m_input_size);
  shape.push_back(m_output_size);
  shape.push_back(m_kernel_size);
  shape.push_back(m_input_size);
  tensor_shape.push_back(shape);
  if (b_bias) {
    m_weights.resize(2);
    tensor_name.push_back(m_name + ".bias");
    tensor_size.push_back(m_output_size);
    shape.resize(1);
    tensor_shape.push_back(shape);
  }

  // get tensors needed by current operator
  std::map<std::string, size_t>::const_iterator it;
  for (size_t i = 0; i < tensor_name.size(); ++i) {
    get_tensor(tensor_name[i], m_name, "Conv1D", m_weights[i]);
  }
  // check consistency of weight tensor's size
  for (size_t i = 0; i < tensor_name.size(); ++i) {
    if (m_weights[i]->size() != tensor_size[i]) {
      LOG(ERROR) << "Size mismatch of tensor [" << m_weights[i]->name()
        << "] between weight file and model file (" << m_weights[i]->size()
        << ", " << tensor_size[i] << ")!" << std::endl;
      return false;
    }
  }
  // set weight tensor's shape
  for (size_t i = 0; i < tensor_name.size(); ++i) {
    m_weights[i]->set_shape(tensor_shape[i]);
  }
  return true;
}

bool Conv1D::reshape(const std::vector<Tensor *> &input,
                     std::vector<Tensor *> &output) {
  if (0 != input[0]->size() % m_input_size) {
    LOG(ERROR) << "Input size [" << input[0]->size() << "] should be divided by ["
      << m_input_size << "]!" << std::endl;
    return false;
  }
  if (2 != input[0]->num_axes()) {
    LOG(ERROR) << "Only support 2d input of shape T*D!" << std::endl;
    return false;
  }
  std::vector<size_t> shape;
  shape.push_back((input[0]->shape(0) + 2*m_padding - m_dilation*(m_kernel_size-1) - 1) / m_stride + 1);
  if (b_bias) {
    m_bias_multiplier.realloc(shape);
    lnn_set(shape[0], 1., m_bias_multiplier.data());
  }
  shape.push_back(m_output_size);
  output[0]->realloc(shape);
  if (m_padding > 0) {
    shape = input[0]->shape();
    shape[0] += 2*m_padding;
    m_buf.realloc(shape);
    lnn_set(m_padding*m_input_size, 0., m_buf.data());
    lnn_copy(input[0]->size(), input[0]->data(), m_buf.data() + m_padding*m_input_size);
    lnn_set(m_padding*m_input_size, 0., m_buf.data() + m_padding*m_input_size + input[0]->size());
  }
  if (m_dilation > 1) {
    shape.resize(1);
    shape[0] = m_kernel_size*m_input_size;
    m_buf_2.realloc(shape);
  }
  return true;
}

void Conv1D::forward_impl(const std::vector<Tensor *> &input,
                          std::vector<Tensor *> &output) {
  size_t T = input[0]->shape(0);
  size_t O = (T + 2*m_padding - m_dilation*(m_kernel_size-1) - 1) / m_stride + 1;
  if (b_bias) {
    lnn_gemm(CblasNoTrans, CblasNoTrans, O, m_output_size, 1,
             1., m_bias_multiplier.data(), m_weights[1]->data(), 0., output[0]->data());
  } else {
    lnn_set(output[0]->size(), 0., output[0]->data());
  }
  const float* data = input[0]->data();
  if (m_padding > 0) data = m_buf.data();
  if (m_dilation == 1) {
    for (size_t i = 0; i < O; ++i) {
      lnn_gemv(CblasNoTrans, m_output_size, m_kernel_size*m_input_size, 1.,
               m_weights[0]->data(), data + i*m_stride*m_input_size, 1.,
               output[0]->data() + i*m_output_size);
    }
  } else {
    for (size_t i = 0; i < O; ++i) {
      const float* cur = data + i*m_stride*m_input_size;
      for (size_t j = 0; j < m_kernel_size; ++j) {
        lnn_copy(m_input_size, cur + j*m_dilation*m_input_size, m_buf_2.data() + j*m_input_size);
      }
      lnn_gemv(CblasNoTrans, m_output_size, m_kernel_size*m_input_size, 1.,
               m_weights[0]->data(), m_buf_2.data(), 1.,
               output[0]->data() + i*m_output_size);
    }
  }
#ifdef DEBUG
  dump(output, std::cout);
#endif  // DEBUG
}

}  // namespace lnn
