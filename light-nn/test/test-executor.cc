#include "executor.h"

#include "net.h"
#include "tensor.h"

#include "gtest/gtest.h"

TEST(ExecutorTest, Interface) {
  lnn::Net net;
  EXPECT_TRUE(net.load("./data/testdata-4-net-simple.json",
                       "./data/testdata-4-weight-simple.dat"));

  lnn::Executor executor(&net);

  // construct net input
  std::vector<lnn::Tensor> input(1);
  float data[] = {-1., -2., -3., -4., -5.};
  std::vector<size_t> shape(1, 5);
  input[0].set_name("input");
  input[0].set_data(data, shape);

  bool res;
  const std::vector<lnn::Tensor *>& output = executor.execute(input, res);
  EXPECT_TRUE(res);
  EXPECT_EQ(1, output.size());
  EXPECT_EQ(1, output[0]->size());
  EXPECT_NEAR(0.982473, output[0]->data()[0], 1E-6);
  EXPECT_FLOAT_EQ(0.982473, output[0]->data()[0]);

  float data2[] = {
    -10., -3., 0.5, 1.9, -2.,
    0.9, 1.3, -4.7, 6.5, -9.1
  };
  shape.resize(2);
  shape[0] = 2;
  shape[1] = 5;
  input[0].set_data(data2, shape);
  const std::vector<lnn::Tensor *>& output2 = executor.execute(input, res);
  EXPECT_TRUE(res);
  EXPECT_EQ(1, output2.size());
  EXPECT_EQ(2, output2[0]->size());
  EXPECT_NEAR(0.992476, output2[0]->data()[0], 1E-6);
  EXPECT_NEAR(0.999719, output2[0]->data()[1], 1E-6);
}
