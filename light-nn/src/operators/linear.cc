#include "operators/linear.h"
#include "utils/math-functions.h"

namespace lnn {

Linear::Linear(const Json::Value &config) : Operator(config) {
  m_name = config["name"].asString();
  m_input_size = config["param"]["input_size"].asInt();
  m_output_size = config["param"]["output_size"].asInt();
  if (config["param"].isMember("bias")) {
    m_bias = config["param"]["bias"].asBool();
  } else {
    m_bias = true;
  }
}

Linear::~Linear() {
}

bool Linear::set_weight(const std::vector<Tensor> &weights,
                        const std::map<std::string, size_t> &weights_name2id) {
	//test linear
	
  if (m_bias) m_weights.resize(2);
  else m_weights.resize(1);
  std::cout << "m_weight.size of linear:" << m_weights.size() << std::endl;

  // get tensors needed by current operator
  std::map<std::string, size_t>::const_iterator it;
  get_tensor(m_name + ".weight", m_name, "Linear", m_weights[0]);//verctor 容器中装入对应的值
  if (m_bias) {
    get_tensor(m_name + ".bias", m_name, "Linear", m_weights[1]);
  }

  // print size of weight and bias 
  //for (size_t i = 0; i < m_weights.size(); i++)
  //{		
	 // if (m_bias&&i > 0) { std::cout << m_name + ".bias" << std::endl; }
	 // else std::cout << m_name + ".weight" << std::endl;
	 // for (size_t j = 0; j < m_weights[i]->size(); j++)
	 // {
		//  std::cout << m_weights[0]->data()[j] << "\t";
	 // }
	 // std::cout << std::endl;
  //}
  // check consistency of weight tensor's size
  if (m_weights[0]->size() != m_output_size * m_input_size) {
    LOG(ERROR) << "Size mismatch of tensor [" << m_weights[0]->name()
      << "] between weight file and model file (" << m_weights[0]->size()
      << ", " << m_output_size * m_input_size << ")!" << std::endl;
    return false;
  }
  if (m_bias) {
    if (m_weights[1]->size() != m_output_size) {
      LOG(ERROR) << "Size mismatch of tensor [" << m_weights[1]->name()
        << "] between weight file and model file (" << m_weights[1]->size()
        << ", " << m_output_size << ")!" << std::endl;
      return false;
    }
  }
  // set weight tensor's shape
  std::vector<size_t> shape;
  shape.push_back(m_output_size);
  shape.push_back(m_input_size);
  m_weights[0]->set_shape(shape);
  if (m_bias) {
    shape.resize(1);
    m_weights[1]->set_shape(shape);
  }
  return true;
}

bool Linear::reshape(const std::vector<Tensor *> &input,
                     std::vector<Tensor *> &output) {
  if (0 != input[0]->size() % m_input_size) {
    LOG(ERROR) << "Input size [" << input[0]->size() << "] should be divided by ["
      << m_input_size << "]!" << std::endl;
    return false;
  }
  std::vector<size_t> shape;
  shape.push_back(input[0]->size() / m_input_size);
  shape.push_back(m_output_size);
  output[0]->realloc(shape);
  if (m_bias) {
    shape.resize(1);
    m_bias_multiplier.realloc(shape);
    lnn_set(shape[0], 1., m_bias_multiplier.data());
  }
 /* std::cout <<"size of shape:"<< shape.size() << std::endl;
  for (size_t i = 0; i < shape.size(); i++)
  {
	  std::cout <<"shape of "+m_name+" is:"<< shape[i] << std::endl;
  }
  std::cout << "shape of bias :"<<m_bias_multiplier.size() << std::endl;
  for (size_t i = 0; i < m_bias_multiplier.size(); i++)
  {
	  std::cout << "each of :"+m_name+"bias"<<m_bias_multiplier.data()[i] << std::endl;
  }*/
  return true;
}

void Linear::forward_impl(const std::vector<Tensor *> &input,
                          std::vector<Tensor *> &output) {
  int batch_size = input[0]->size() / m_input_size;
  lnn_gemm(CblasNoTrans, CblasTrans, batch_size, m_output_size, m_input_size,
           1., input[0]->data(), m_weights[0]->data(), 0., output[0]->data());
  std::cout << "size of output:"<<output[0]->size() << std::endl;
  std::cout << "shape of output:"<<&output[0]->shape() << std::endl;

  for (size_t i = 0; i < m_bias_multiplier.size(); i++)
  {
	  std::cout << m_bias_multiplier.data()[i]<< std::endl;
  }
  if (m_bias) {
    lnn_gemm(CblasNoTrans, CblasNoTrans, batch_size, m_output_size, 1,
             1., m_bias_multiplier.data(), m_weights[1]->data(), 1., output[0]->data());
  }
#ifdef DEBUG
  dump(output, std::cout);
#endif  // DEBUG
}

}  // namespace lnn
