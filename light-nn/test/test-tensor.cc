#include "tensor.h"

#include "gtest/gtest.h"

TEST(TensorTest, Basic) {
  float *data = new float[12];
  std::vector<size_t> shape;
  shape.push_back(3);
  shape.push_back(4);

  lnn::Tensor tensor("tensor-4-test");
  tensor.set_data(data, shape);

  EXPECT_EQ(12, tensor.size());
  EXPECT_EQ(2, tensor.num_axes());
  EXPECT_EQ(3, tensor.shape(0));
  EXPECT_EQ(4, tensor.shape(1));
  EXPECT_EQ(3, tensor.shape(-2));
  EXPECT_EQ(4, tensor.shape(-1));
  EXPECT_EQ(3, tensor.shape(2));
  EXPECT_EQ(4, tensor.shape(-3));
  EXPECT_EQ(4, tensor.count(1));
  EXPECT_EQ(12, tensor.count(0));
  EXPECT_STREQ("tensor-4-test", tensor.name().c_str());

  delete [] data;
  data = NULL;
}

TEST(TensorTest, Advanced) {
  float *data = new float[12];
  std::vector<size_t> shape;
  shape.push_back(3);
  shape.push_back(4);

  lnn::Tensor tensor("tensor-4-test");
  tensor.set_data(data, 12);
  tensor.set_shape(shape);

  EXPECT_EQ(12, tensor.size());
  EXPECT_EQ(2, tensor.num_axes());
  EXPECT_EQ(3, tensor.shape(0));
  EXPECT_EQ(4, tensor.shape(1));
  EXPECT_EQ(3, tensor.shape(-2));
  EXPECT_EQ(4, tensor.shape(-1));
  EXPECT_EQ(3, tensor.shape(2));
  EXPECT_EQ(4, tensor.shape(-3));
  EXPECT_STREQ("tensor-4-test", tensor.name().c_str());

  std::vector<size_t> new_shape(1, 12);
  EXPECT_TRUE(tensor.reshape(new_shape));
  EXPECT_EQ(1, tensor.num_axes());
  EXPECT_EQ(12, tensor.shape(0));
  EXPECT_EQ(12, tensor.size());

  new_shape[0] = 15;
  EXPECT_FALSE(tensor.reshape(new_shape));
  tensor.realloc(new_shape);
  EXPECT_EQ(15, tensor.size());
  EXPECT_EQ(1, tensor.num_axes());
  EXPECT_EQ(15, tensor.shape(0));

  delete [] data;
  data = NULL;
}
