#include "operators/pooling1d.h"
#include "utils/math-functions.h"

#include <float.h>

namespace lnn {

Pooling1D::Pooling1D(const Json::Value &config) : Operator(config) {
  m_name = config["name"].asString();
  m_input_size = config["param"]["input_size"].asInt();
  if (config["param"].isMember("global_pooling")) {
    b_global_pooling = config["param"]["global_pooling"].asBool();
  } else {
    b_global_pooling = false;
  }
  if (!b_global_pooling) {
    m_kernel_size = config["param"]["kernel_size"].asInt();
    if (config["param"].isMember("stride")) {
      m_stride = config["param"]["stride"].asInt();
    } else {
      m_stride = 1;
    }
    if (config["param"].isMember("dilation")) {
      m_dilation = config["param"]["dilation"].asInt();
    } else {
      m_dilation = 1;
    }
  }
  if (config["param"].isMember("padding")) {
    m_padding = config["param"]["padding"].asInt();
  } else {
    m_padding = 0;
  }
  if (config["param"].isMember("type")) {
    m_type = config["param"]["type"].asString();
  } else {
    m_type = "MAX";
  }
}

Pooling1D::~Pooling1D() {
}

bool Pooling1D::set_weight(const std::vector<Tensor> &weights,
                           const std::map<std::string, size_t> &weights_name2id) {
  return true;
}

bool Pooling1D::reshape(const std::vector<Tensor *> &input,
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
  if (!b_global_pooling) {
    shape.push_back((input[0]->shape(0) + 2*m_padding - m_dilation*(m_kernel_size-1) - 1) / m_stride + 1);
  }
  shape.push_back(input[0]->shape(1));
  output[0]->realloc(shape);
  if (m_padding > 0) {
    shape = input[0]->shape();
    shape[0] += 2*m_padding;
    m_buf.realloc(shape);
    lnn_set(m_padding*m_input_size, 0., m_buf.data());
    lnn_copy(input[0]->size(), input[0]->data(), m_buf.data() + m_padding*m_input_size);
    lnn_set(m_padding*m_input_size, 0., m_buf.data() + m_padding*m_input_size + input[0]->size());
  }
  return true;
}

void Pooling1D::forward_impl(const std::vector<Tensor *> &input,
                             std::vector<Tensor *> &output) {
  if ("MAX" == m_type) {
    forward_max(input, output);
  } else {  // "MEAN" == m_type
    forward_mean(input, output);
  }
//  for (size_t i = 0; i < output[0]->size(); ++i) {
//    if (i % output[0]->shape(1) == 0) std::cout << std::endl;
//    std::cout << std::fixed << std::setprecision(6) << output[0]->data()[i] << ' ';
//  }
//  std::cout << std::endl;
}

void Pooling1D::forward_max(const std::vector<Tensor *> &input,
                            std::vector<Tensor *> &output) {
  size_t T = input[0]->shape(0);
  size_t D = input[0]->shape(1);
  const float* data = input[0]->data();
  if (m_padding > 0) data = m_buf.data();
  float* o_data = output[0]->data();
  if (b_global_pooling) {
    size_t nrow = m_padding > 0 ? T+2*m_padding : T;
    for (size_t i = 0; i < D; ++i) {
      o_data[i] = -FLT_MAX;
      for (size_t j = 0; j < nrow; ++j) {
        if (*(data+j*D+i) > o_data[i]) {
          o_data[i] = *(data+j*D+i);
        }
      }
    }
  } else {
    size_t O = (T + 2*m_padding - m_dilation*(m_kernel_size-1) - 1) / m_stride + 1;
    for (size_t i = 0; i < O; ++i) {
      const float* cur = data + i*m_stride*D;
      for (size_t j = 0; j < D; ++j) {
        *(o_data+i*D+j) = -FLT_MAX;
        for (size_t k = 0; k < m_kernel_size; ++k) {
          if (*(cur + k*m_dilation*D + j) > *(o_data+i*D+j)) {
            *(o_data+i*D+j) = *(cur + k*m_dilation*D + j);
          }
        }
      }
    }
  }
}

void Pooling1D::forward_mean(const std::vector<Tensor *> &input,
                             std::vector<Tensor *> &output) {
  size_t T = input[0]->shape(0);
  size_t D = input[0]->shape(1);
  const float* data = input[0]->data();
  if (m_padding > 0) data = m_buf.data();
  float* o_data = output[0]->data();
  if (b_global_pooling) {
    size_t nrow = m_padding > 0 ? T+2*m_padding : T;
    lnn_copy(D, data, o_data);
    for (size_t i = 1; i < nrow; ++i) {
      lnn_axpy(D, 1., data+i*D, o_data);
    }
    lnn_scal(D, 1./nrow, o_data);
  } else {
    size_t O = (T + 2*m_padding - m_dilation*(m_kernel_size-1) - 1) / m_stride + 1;
    for (size_t i = 0; i < O; ++i) {
      const float* cur = data + i*m_stride*D;
      lnn_copy(D, cur, o_data+i*D);
      for (size_t j = 1; j < m_kernel_size; ++j) {
        lnn_axpy(D, 1., cur+j*m_dilation*D, o_data+i*D);
      }
      lnn_scal(D, 1./m_kernel_size, o_data+i*D);
    }
  }
#ifdef DEBUG
  dump(output, std::cout);
#endif  // DEBUG
}

}  // namespace lnn
