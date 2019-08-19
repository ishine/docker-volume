#include "cblas.h"
#include "executor.h"

#include "config.h"
#include "net.h"
#include "operator-factory.h"


#ifdef USE_OPENMP
#include <omp.h>
#endif  // USE_OPENMP

namespace lnn {

//构造函数
Executor::Executor(const Net* net, int num_threads) {
  LOG(INFO) << "Version: " << LNN_VERSION_MAJOR << "." << LNN_VERSION_MINOR << std::endl;
  m_net = const_cast<Net*>(net);
  m_num_threads = num_threads;
  // initialize operators
  const Json::Value *net_json = m_net->json();
  int op_num = (*net_json)["operators"].size();
  m_operators.resize(op_num);
  LOG(INFO) << "#op: " << op_num << std::endl;
  const std::map<std::string, size_t>& name2id = m_net->weight_tensor_name2id();
  for (std::map<std::string, size_t>::const_iterator it = name2id.begin();
       it != name2id.end(); ++it) {
    LOG(INFO) << it->first << " -> " << it->second << std::endl;
  }
  for (int i = 0; i < op_num; ++i) {
    std::string type = (*net_json)["operators"][i]["type"].asString();
    get_operator(type, (*net_json)["operators"][i], m_operators[i]);
    m_operators[i]->set_weight(m_net->weight_tensors(), m_net->weight_tensor_name2id());
  }
  // initialize dynamic tensors
  const std::vector<std::string> &dynamic_tensor_name =
    m_net->dynamic_tensor_name();
  m_dynamic_tensors.resize(dynamic_tensor_name.size());
  for (size_t i = 0; i < m_dynamic_tensors.size(); ++i) {
    m_dynamic_tensors[i].set_name(dynamic_tensor_name[i]);
  }
  // figure out output tensors
  const std::vector<std::vector<size_t> > &output_ids = m_net->op_output_ids();
  size_t output_tensor_num = output_ids[output_ids.size()-1].size();
  m_output_tensors.resize(output_tensor_num);
  for (size_t i = 0; i < output_tensor_num; ++i) {
    size_t id = output_ids[output_ids.size()-1][i];
    m_output_tensors[i] = &(m_dynamic_tensors[id]);
  }
#ifdef USE_OPENMP
  omp_set_dynamic(0);
  omp_set_num_threads(m_num_threads);
#endif  // USE_OPENMP
  openblas_set_num_threads(m_num_threads);
}


//析构函数
Executor::~Executor() {
  m_net = NULL;
  m_dynamic_tensors.clear();
  for (size_t i = 0; i < m_operators.size(); ++i) {
    delete m_operators[i];
    m_operators[i] = NULL;
  }
  m_operators.clear();
  m_output_tensors.clear();
}

//前向传播
const std::vector<Tensor *> & Executor::execute(
  const std::vector<Tensor> &input, bool &success) {
  success = true;
  // initialize net input
  const std::map<std::string, size_t> &tensor_name2id = m_net->dynamic_tensor_name2id();
  std::map<std::string, size_t>::const_iterator it;
  for (size_t i = 0; i < input.size(); ++i) {
    it = tensor_name2id.find(const_cast<Tensor &>(input[i]).name());
    m_dynamic_tensors[it->second] = input[i];
  }
  // execute each operator's forward
  const std::vector<std::vector<size_t> > &input_ids = m_net->op_input_ids();
  const std::vector<std::vector<size_t> > &output_ids = m_net->op_output_ids();
  std::vector<Tensor *> op_input, op_output; //存放的tensor类的tensor
  //这里查看网络的op向前传播的过程，单个op的shape，size
  //填充op_output的数量，具体每个op的shape和size在reshape中realloc
  for (size_t i = 0; i < m_operators.size(); ++i) {
    op_input.resize(input_ids[i].size());
    for (size_t j = 0; j < input_ids[i].size(); ++j) {
      op_input[j] = &(m_dynamic_tensors[input_ids[i][j]]);
    }
    op_output.resize(output_ids[i].size());
    for (size_t j = 0; j < output_ids[i].size(); ++j) {
      op_output[j] = &(m_dynamic_tensors[output_ids[i][j]]);
    }
    if (! m_operators[i]->forward(op_input, op_output)) {
	  LOG(INFO) << "forward failed!" << std::endl;
      success = false;
      break;  
    }
	//reshape
	if (i = m_operators.size() - 2)
	{
		std::vector<size_t>shape;
		shape.push_back(1);
		shape.push_back(op_output[0]->shape(0)*op_output[0]->shape(1)*op_output[0]->shape(2));
		op_output[0]->realloc(shape);
	}

	//输出每一层的结果，便于检查
	for (size_t n = 0; n < output_ids[i].size(); n++)
	{
		LOG(INFO) <<"第"<<i<<"层"<<"第"<<n<<"特征图输出："<< op_output[n]->name()<<":\t";
		for (size_t m = 0; m < op_output[n]->shape().size(); m++)
		{
			std::cout << op_output[n]->shape(m) << "  ";
		}
	}
	std::cout << std::endl;
	//输出每一层特征图的值，进行对比
	/*for (size_t n = 0; n < output_ids[i].size(); n++)
	{
		for (size_t c = 0; c < op_output[n]->shape(0); ++c) {
			for (size_t w = 0; w < op_output[n]->shape(1); w++)
			{
				for (size_t h = 0; h < op_output[n]->shape(2); h++)
				{
					std::cout << std::fixed << std::setprecision(4)
						<< op_output[n]->data()[c*op_output[n]->shape(1)*op_output[n]->shape(2) + w * op_output[n]->shape(2) + h] << ',';
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;*/
  }
  return m_output_tensors;
}

}  // namespace lnn
