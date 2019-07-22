#include <iomanip>
#include <iostream>
#include "light-nn.h"

int main(int argc, char* argv[]) {
  if (3 != argc) {
    std::cerr << "usage: " << argv[0] << " net-file weight-file" << std::endl;
    return -1;
  }

  // create net (it should be created only once in the main process)
  lnn::Net net;
  if (!net.load(argv[1], argv[2])) {
    std::cerr << "load net-file & weight-file failed!" << std::endl;
    return -1;
  }

  // create executor (it can be created once for each thread)
  lnn::Executor executor(&net);

  // construct net input
  std::vector<lnn::Tensor> input(1);
  float data[] = {-1., -2., -3., -4., -5.};
  std::vector<size_t> shape(1, 5);
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
