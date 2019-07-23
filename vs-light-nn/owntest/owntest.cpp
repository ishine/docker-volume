#include <iomanip>
#include <iostream>
#include "light-nn.h"



//int argc, char* argv[]
int main() {
	/*if (3 != argc) {
		std::cerr << "usage: " << argv[0] << " net-file weight-file" << std::endl;
		return -1;
	}*/

	// create net (it should be created only once in the main process) argv[1], argv[2]
	const char *model_file; // 加入const后 字符串指针可以修改，其中的元素不可以修改
	const char *weight_file;
	model_file = "D:\\docker-volume\\vs-light-nn\\x64\\Debug\\conv1.json";
	//weight_file = "D:\\docker-volume\\vs-light-nn\\x64\\Debug\\weight-ljf.dat";
	weight_file = "D:\\docker-volume\\vs-light-nn\\x64\\Debug\\weight-conv1.dat";
	lnn::Net net;
	if (!net.load(model_file,weight_file)) {
		std::cerr << "load net-file & weight-file failed!" << std::endl;
		return -1;
	}

	// create executor (it can be created once for each thread)
	lnn::Executor executor(&net);
	
	// construct net input
	std::vector<lnn::Tensor> input(1);
	float data[] = { -1., -2., -3., -4., -5.,
					1.,2.,3.,4.,5.,
					 -1., -2., -3., -4., -5.,
					1.,2.,3.,4.,5.,
					1.,2.,3.,4.,5. };
	std::vector<size_t> shape(2);
	shape[0] = 5;
	shape[1] = 5;
	input[0].set_name("input");
	input[0].set_data(data, shape);

	// compute and get the result
	bool res;
	const std::vector<lnn::Tensor *>& output = executor.execute(input, res);
	for (size_t i = 0; i < output[0]->size(); ++i) {
		std::cout << std::fixed << std::setprecision(6)
			<< output[0]->data()[i] << ',';
	}
	std::cout << std::endl;

	return 0;
}
