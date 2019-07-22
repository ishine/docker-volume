#include "operators/lstm.h"
#include "utils/math-functions.h"

namespace lnn {

LSTM::LSTM(const Json::Value &config) : Operator(config) {
  m_name = config["name"].asString();
  m_input_size = config["param"]["input_size"].asInt();
  m_output_size = config["param"]["output_size"].asInt();
  if (config["param"].isMember("bias")) {
    b_bias = config["param"]["bias"].asBool();
  } else {
    b_bias = true;
  }
  if (config["param"].isMember("bidirectional")) {
    b_bidirectional = config["param"]["bidirectional"].asBool();
  } else {
    b_bidirectional = true;
  }
  if (config["param"].isMember("topK")) {
    m_top_k = config["param"]["topK"].asInt();
    b_with_cache = true;
  } else {
    b_with_cache = false;
  }
  m_cache = m_cache_reverse = NULL;
  b_cached = b_cached_reverse = NULL;
}

LSTM::~LSTM() {
  if (NULL != m_cache) {
    delete [] m_cache;
    m_cache = NULL;
  }
  if (NULL != m_cache_reverse) {
    delete [] m_cache_reverse;
    m_cache_reverse = NULL;
  }
  if (NULL != b_cached) {
    delete [] b_cached;
    b_cached = NULL;
  }
  if (NULL != b_cached_reverse) {
    delete [] b_cached_reverse;
    b_cached_reverse = NULL;
  }
}

bool LSTM::set_weight(const std::vector<Tensor> &weights,
                      const std::map<std::string, size_t> &weights_name2id) {
  std::vector<std::string> tensor_name;
  std::vector<size_t> tensor_size, shape;
  std::vector<std::vector<size_t> > tensor_shape;
  if (b_bidirectional) {
    if (b_bias) {
      m_weights.resize(6);
      tensor_name.push_back(m_name + ".w_ih");
      tensor_name.push_back(m_name + ".w_hh");
      tensor_name.push_back(m_name + ".b");
      tensor_name.push_back(m_name + ".w_ih_reverse");
      tensor_name.push_back(m_name + ".w_hh_reverse");
      tensor_name.push_back(m_name + ".b_reverse");
      tensor_size.push_back(4 * m_output_size * m_input_size);
      tensor_size.push_back(4 * m_output_size * m_output_size);
      tensor_size.push_back(4 * m_output_size);
      tensor_size.push_back(4 * m_output_size * m_input_size);
      tensor_size.push_back(4 * m_output_size * m_output_size);
      tensor_size.push_back(4 * m_output_size);
      shape.push_back(4 * m_output_size);
      shape.push_back(m_input_size);
      tensor_shape.push_back(shape);
      shape[1] = m_output_size;
      tensor_shape.push_back(shape);
      shape.resize(1);
      tensor_shape.push_back(shape);
      shape.push_back(m_input_size);
      tensor_shape.push_back(shape);
      shape[1] = m_output_size;
      tensor_shape.push_back(shape);
      shape.resize(1);
      tensor_shape.push_back(shape);
    } else {
      m_weights.resize(4);
      tensor_name.push_back(m_name + ".w_ih");
      tensor_name.push_back(m_name + ".w_hh");
      tensor_name.push_back(m_name + ".w_ih_reverse");
      tensor_name.push_back(m_name + ".w_hh_reverse");
      tensor_size.push_back(4 * m_output_size * m_input_size);
      tensor_size.push_back(4 * m_output_size * m_output_size);
      tensor_size.push_back(4 * m_output_size * m_input_size);
      tensor_size.push_back(4 * m_output_size * m_output_size);
      shape.push_back(4 * m_output_size);
      shape.push_back(m_input_size);
      tensor_shape.push_back(shape);
      shape[1] = m_output_size;
      tensor_shape.push_back(shape);
      shape[1] = m_input_size;
      tensor_shape.push_back(shape);
      shape[1] = m_output_size;
      tensor_shape.push_back(shape);
    }
  } else {
    if (b_bias) {
      m_weights.resize(3);
      tensor_name.push_back(m_name + ".w_ih");
      tensor_name.push_back(m_name + ".w_hh");
      tensor_name.push_back(m_name + ".b");
      tensor_size.push_back(4 * m_output_size * m_input_size);
      tensor_size.push_back(4 * m_output_size * m_output_size);
      tensor_size.push_back(4 * m_output_size);
      shape.push_back(4 * m_output_size);
      shape.push_back(m_input_size);
      tensor_shape.push_back(shape);
      shape[1] = m_output_size;
      tensor_shape.push_back(shape);
      shape.resize(1);
      tensor_shape.push_back(shape);
    } else {
      m_weights.resize(2);
      tensor_name.push_back(m_name + ".w_ih");
      tensor_name.push_back(m_name + ".w_hh");
      tensor_size.push_back(4 * m_output_size * m_input_size);
      tensor_size.push_back(4 * m_output_size * m_output_size);
      shape.push_back(4 * m_output_size);
      shape.push_back(m_input_size);
      tensor_shape.push_back(shape);
      shape[1] = m_output_size;
      tensor_shape.push_back(shape);
    }
  }

  // get tensors needed by current operator
  std::map<std::string, size_t>::const_iterator it;
  for (size_t i = 0; i < tensor_name.size(); ++i) {
    get_tensor(tensor_name[i], m_name, "LSTM", m_weights[i]);
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

  // set cache info
  if (b_with_cache) {
    if (m_top_k <= 0) {
      LOG(ERROR) << "TopK should be in [1, vocabulary_size) interval!" << std::endl;
      return false;
    }
    size_t num = 4 * m_output_size * m_top_k;
    m_cache = new float[num];
    b_cached = new bool[m_top_k];
    lnn_set(num, 0., m_cache);
    if (b_bidirectional) {
      m_cache_reverse = new float[num];
      b_cached_reverse = new bool[m_top_k];
      lnn_set(num, 0., m_cache_reverse);
    }
    for (size_t i = 0; i < m_top_k; ++i) {
      b_cached[i] = false;
      if (b_bidirectional) b_cached_reverse[i] = false;
    }
  }
  return true;
}

bool LSTM::reshape(const std::vector<Tensor *> &input,
                   std::vector<Tensor *> &output) {
  if (0 != input[0]->size() % m_input_size) {
    LOG(ERROR) << "Input size [" << input[0]->size() << "] should be divided by ["
      << m_input_size << "]!" << std::endl;
    return false;
  }
  if (2 != input[0]->num_axes()) {
    LOG(ERROR) << "Only support 2d input of shape T*D of the first input!" << std::endl;
    return false;
  }
  if (b_with_cache) {
    if (2 != input.size()) {
      LOG(ERROR) << "If you set topK in json file, there must have two input tensors!" << std::endl;
      return false;
    }
    if (1 != input[1]->num_axes() || input[1]->shape(0) != input[0]->shape(0)) {
      LOG(ERROR) << "The first axis of the two input tensors should be the same!" << std::endl;
      return false;
    }
  }
  std::vector<size_t> shape(input[0]->shape());
  shape[1] = b_bidirectional ? 2 * m_output_size : m_output_size;
  output[0]->realloc(shape);
  m_cell.realloc(shape);
  shape[1] = 4 * m_output_size;
  m_buf.realloc(shape);
  lnn_set(m_buf.size(), 0., m_buf.data());
  if (b_bidirectional) {
    m_buf_reverse.realloc(shape);
    lnn_set(m_buf_reverse.size(), 0., m_buf_reverse.data());
  }
  shape.resize(1);
  shape[0] = b_bidirectional ? 2 * m_output_size : m_output_size;
  m_h_0.realloc(shape);
  m_c_0.realloc(shape);
  lnn_set(m_h_0.size(), 0., m_h_0.data());
  lnn_set(m_c_0.size(), 0., m_c_0.data());
  if (b_bias) {
    shape[0] = input[0]->shape(0);
    m_bias_multiplier.realloc(shape);
    lnn_set(shape[0], 1., m_bias_multiplier.data());
  }
  shape[0] = m_output_size;
  m_ig.realloc(shape);
  return true;
}

void LSTM::forward_impl(const std::vector<Tensor *> &input,
                        std::vector<Tensor *> &output) {
  size_t T = input[0]->shape(0);
  size_t D = input[0]->shape(1);
  if (b_bias) {
    lnn_gemm(CblasNoTrans, CblasNoTrans, T, 4*m_output_size, 1,
             1., m_bias_multiplier.data(), m_weights[2]->data(), 0., m_buf.data());
    if (b_bidirectional) {
      lnn_gemm(CblasNoTrans, CblasNoTrans, T, 4*m_output_size, 1,
               1., m_bias_multiplier.data(), m_weights[5]->data(), 0., m_buf_reverse.data());
    }
  } else {
    lnn_set(m_buf.size(), 0., m_buf.data());
    lnn_set(m_buf_reverse.size(), 0., m_buf_reverse.data());
  }
  // forward
  if (!b_with_cache) {
    lnn_gemm(CblasNoTrans, CblasTrans, T, 4*m_output_size, D,
             1., input[0]->data(), m_weights[0]->data(), 1., m_buf.data());
  }
  for (size_t t = 1; t <= T; ++t) {
    size_t offset = (t-1)*4*m_output_size;
    if (b_with_cache) {
      size_t idx = size_t(input[1]->data()[t-1]);
      if (idx >= m_top_k) {
        lnn_gemv(CblasNoTrans, 4*m_output_size, D, 1., m_weights[0]->data(),
                 input[0]->data() + (t-1)*D, 1., m_buf.data() + offset);
      } else {
        if (b_cached[idx]) {
          lnn_add(4*m_output_size, m_buf.data() + offset,
                  m_cache + idx*4*m_output_size, m_buf.data() + offset);
        } else {
          lnn_gemv(CblasNoTrans, 4*m_output_size, D, 1., m_weights[0]->data(),
                   input[0]->data() + (t-1)*D, 0., m_cache + idx*4*m_output_size);
          lnn_add(4*m_output_size, m_buf.data() + offset,
                  m_cache + idx*4*m_output_size, m_buf.data() + offset);
          b_cached[idx] = true;
        }
      }
    }
    size_t step = b_bidirectional ? 2*m_output_size : m_output_size;
    const float* h_t_1 = (1 == t) ? m_h_0.data() : output[0]->data() + (t-2)*step;
    const float* c_t_1 = (1 == t) ? m_c_0.data() : m_cell.data() + (t-2)*step;
    lnn_gemv(CblasNoTrans, 4*m_output_size, m_output_size, 1., m_weights[1]->data(),
             h_t_1, 1., m_buf.data() + offset);
    lnn_sigmoid(3*m_output_size, m_buf.data() + offset, m_buf.data() + offset);
    lnn_tanh(m_output_size, m_buf.data() + offset + 3*m_output_size,
             m_buf.data() + offset + 3*m_output_size);
    const float* i_t = m_buf.data() + offset;
    const float* f_t = m_buf.data() + offset + m_output_size;
    const float* o_t = m_buf.data() + offset + 2*m_output_size;
    const float* g_t = m_buf.data() + offset + 3*m_output_size;
    float* c_t = m_cell.data() + (t-1)*step;
    float* ig = m_ig.data();
    float* h_t = output[0]->data() + (t-1)*step;
    lnn_mul(m_output_size, f_t, c_t_1, c_t);
    lnn_mul(m_output_size, i_t, g_t, ig);
    lnn_add(m_output_size, c_t, ig, c_t);
    lnn_tanh(m_output_size, c_t, ig);
    lnn_mul(m_output_size, o_t, ig, h_t);
  }
  // backward
  if (b_bidirectional) {
    int w_xh_id = b_bias ? 3 : 2;
    int w_hh_id = b_bias ? 4 : 3;
    if (!b_with_cache) {
      lnn_gemm(CblasNoTrans, CblasTrans, T, 4*m_output_size, D,
               1., input[0]->data(), m_weights[w_xh_id]->data(), 1., m_buf_reverse.data());
    }
    for (size_t t = T; t >= 1; --t) {
      size_t offset = (t-1)*4*m_output_size;
      if (b_with_cache) {
        size_t idx = size_t(input[1]->data()[t-1]);
        if (idx >= m_top_k) {
          lnn_gemv(CblasNoTrans, 4*m_output_size, D, 1., m_weights[w_xh_id]->data(),
                   input[0]->data() + (t-1)*D, 1., m_buf_reverse.data() + offset);
        } else {
          if (b_cached_reverse[idx]) {
            lnn_add(4*m_output_size, m_buf_reverse.data() + offset,
                    m_cache_reverse + idx*4*m_output_size, m_buf_reverse.data() + offset);
          } else {
            lnn_gemv(CblasNoTrans, 4*m_output_size, D, 1., m_weights[w_xh_id]->data(),
                     input[0]->data() + (t-1)*D, 0., m_cache_reverse + idx*4*m_output_size);
            lnn_add(4*m_output_size, m_buf_reverse.data() + offset,
                    m_cache_reverse + idx*4*m_output_size, m_buf_reverse.data() + offset);
            b_cached_reverse[idx] = true;
          }
        }
      }
      size_t step = b_bidirectional ? 2*m_output_size : m_output_size;
      const float* h_t_1 = (T == t) ? m_h_0.data() : output[0]->data() + t*step + m_output_size;
      const float* c_t_1 = (T == t) ? m_c_0.data() : m_cell.data() + t*step + m_output_size;
      lnn_gemv(CblasNoTrans, 4*m_output_size, m_output_size, 1., m_weights[w_hh_id]->data(),
               h_t_1, 1., m_buf_reverse.data() + offset);
      lnn_sigmoid(3*m_output_size, m_buf_reverse.data() + offset,
                  m_buf_reverse.data() + offset);
      lnn_tanh(m_output_size, m_buf_reverse.data() + offset + 3*m_output_size,
               m_buf_reverse.data() + offset + 3*m_output_size);
      const float* i_t = m_buf_reverse.data() + offset;
      const float* f_t = m_buf_reverse.data() + offset + m_output_size;
      const float* o_t = m_buf_reverse.data() + offset + 2*m_output_size;
      const float* g_t = m_buf_reverse.data() + offset + 3*m_output_size;
      float* c_t = m_cell.data() + (t-1)*step + m_output_size;
      float* ig = m_ig.data();
      float* h_t = output[0]->data() + (t-1)*step + m_output_size;
      lnn_mul(m_output_size, f_t, c_t_1, c_t);
      lnn_mul(m_output_size, i_t, g_t, ig);
      lnn_add(m_output_size, c_t, ig, c_t);
      lnn_tanh(m_output_size, c_t, ig);
      lnn_mul(m_output_size, o_t, ig, h_t);
    }
  }
#ifdef DEBUG
  dump(output, std::cout);
#endif  // DEBUG
}

}  // namespace lnn
