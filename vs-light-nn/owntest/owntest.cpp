#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include "light-nn.h"
#include "tensor.h"

// TODO:加入bn，pooling2d和leakyrelu三个op
// TODO:内联函数(引用)，cout输出类型，类的声明与定义。完整的类使用。头文件+cpp
// TODO:blas矩阵计算 conv2dC++实现
//int argc, char* argv[]
 int main() {
	/*if (3 != argc) {
		std::cerr << "usage: " << argv[0] << " net-file weight-file" << std::endl;
		return -1;
	}*/

	// create net (it should be created only once in the main process) argv[1], argv[2]
	const char *model_file; // 加入const后 字符串指针可以修改，其中的元素不可以修改
	const char *weight_file;
	model_file = "D:\\docker-volume\\vs-light-nn\\x64\\Debug\\demo-conv2d.json";
	//weight_file = "D:\\docker-volume\\vs-light-nn\\x64\\Debug\\conv1d.dat";
	weight_file = "D:\\docker-volume\\vs-light-nn\\x64\\Debug\\conv2d.dat";
	lnn::Net net;
	if (!net.load(model_file,weight_file)) {
		std::cerr << "load net-file & weight-file failed!" << std::endl;
		return -1;
	}
	//m_weight_tensor是一个vector，其中的元素为tensor类
	/*std::vector<lnn::Tensor> templist = net.weight_tensors();
	for (size_t i = 0; i < templist.size(); i++)
	{

		for (size_t j = 0; j < templist[i].size(); j++)
		{
			std::cout << templist[i].data()[j] << std::endl;
		}
	}*/

	// create executor (it can be created once for each thread)
	// init executor
	lnn::Executor executor(&net);
	
	// construct net input
	std::vector<lnn::Tensor> input(1);
	float data[] = { 1.,2.,3.,4.,5.,
					 -1., -2., -3., -4., -5.,
					1.,2.,3.,4.,5.,
					-1., -2., -3., -4., -5.,
					1.,2.,3.,4.,5., 
					1.,2.,3.,4.,5.,
					 -1., -2., -3., -4., -5.,
					1.,2.,3.,4.,5.,
					-1., -2., -3., -4., -5.,
					1.,2.,3.,4.,5.,  
					1.,2.,3.,4.,5.,
					 -1., -2., -3., -4., -5.,
					1.,2.,3.,4.,5.,
					-1., -2., -3., -4., -5.,
					1.,2.,3.,4.,5.,  };
	std::vector<size_t> shape(3);
	shape[0] = 3;
	shape[1] = 5;
	shape[2] = 5;
	input[0].set_name("input");
	input[0].set_data(data, shape);
	std::cout << input[0].shape(0) << std::endl;
	/*for (int i = 0; i < input[0].size(); i++)
	{
		std::cout << *(input[0].data() + i) << std::endl;
	}
	std::cout << input[0].count(0,2) << std::endl;*/
	/*std::cout << input[0].name() << std::endl;*/

	// compute and get the result
	bool res;
	const std::vector<lnn::Tensor *>& output = executor.execute(input, res);
	//std::cout << output[0]->size() << std::endl;
	for (size_t i = 0; i < output[0]->size(); ++i) {
		std::cout << std::fixed << std::setprecision(6)
			<< output[0]->data()[i] << ',';
	}
	std::cout << std::endl;

	return 0;
}
