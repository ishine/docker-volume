#include "operators/concat.h"

#include "gtest/gtest.h"

class ConcatTest : public testing::Test {
 protected:
  virtual void SetUp() {
    // construct json config
    m_config["name"] = "concat1";
    Json::Value param;
    param["axis"] = -1;
    m_config["param"] = param;

    // construct input & output tensor vector
    m_input_tensor = new lnn::Tensor;
    m_input_tensor_2 = new lnn::Tensor;
    m_output_tensor = new lnn::Tensor;
    m_input_vec.push_back(m_input_tensor);
    m_input_vec.push_back(m_input_tensor_2);
    m_output_vec.push_back(m_output_tensor);
  }
  virtual void TearDown() {
    delete m_input_tensor;
    delete m_input_tensor_2;
    delete m_output_tensor;
  }

  Json::Value m_config;
  lnn::Tensor *m_input_tensor;
  lnn::Tensor *m_input_tensor_2;
  lnn::Tensor *m_output_tensor;
  std::vector<lnn::Tensor *> m_input_vec;
  std::vector<lnn::Tensor *> m_output_vec;
};

TEST_F(ConcatTest, 2DLastAxis) {
  m_config["param"]["axis"] = -1;
  lnn::Concat concat(m_config);

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 5;
  shape[1] = 3;
  m_input_tensor->set_data(input_data, shape);
  float input_data2[10];
  for (int i = 0; i < 10; ++i) { input_data2[i] = i; };
  shape[1] = 2;
  m_input_tensor_2->set_data(input_data2, shape);

  EXPECT_TRUE(concat.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(25, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(5, m_output_vec[0]->shape(0));
  EXPECT_EQ(5, m_output_vec[0]->shape(1));
  float target[] = {
    0.,  1.,  2.,  0., 1.,
    3.,  4.,  5.,  2., 3.,
    6.,  7.,  8.,  4., 5.,
    9.,  10., 11., 6., 7.,
    12., 13., 14., 8., 9.
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(ConcatTest, 2DFirstAxis) {
  m_config["param"]["axis"] = 0;
  lnn::Concat concat(m_config);

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 5;
  shape[1] = 3;
  m_input_tensor->set_data(input_data, shape);
  float input_data2[6];
  for (int i = 0; i < 6; ++i) { input_data2[i] = i; };
  shape[0] = 2;
  m_input_tensor_2->set_data(input_data2, shape);

  EXPECT_TRUE(concat.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(21, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(7, m_output_vec[0]->shape(0));
  EXPECT_EQ(3, m_output_vec[0]->shape(1));
  float target[] = {
    0.,  1.,  2.,
    3.,  4.,  5.,
    6.,  7.,  8.,
    9.,  10., 11.,
    12., 13., 14.,
    0., 1., 2.,
    3., 4., 5.
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(ConcatTest, 3DMiddleAxis) {
  m_config["param"]["axis"] = 1;
  lnn::Concat concat(m_config);

  float input_data[24];
  for (int i = 0; i < 24; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(3);
  shape[0] = 2;
  shape[1] = 3;
  shape[2] = 4;
  m_input_tensor->set_data(input_data, shape);
  float input_data2[16];
  for (int i = 0; i < 16; ++i) { input_data2[i] = i; };
  shape[1] = 2;
  m_input_tensor_2->set_data(input_data2, shape);

  EXPECT_TRUE(concat.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(40, m_output_vec[0]->size());
  EXPECT_EQ(3, m_output_vec[0]->num_axes());
  EXPECT_EQ(2, m_output_vec[0]->shape(0));
  EXPECT_EQ(5, m_output_vec[0]->shape(1));
  EXPECT_EQ(4, m_output_vec[0]->shape(2));
  float target[] = {
    0.,  1.,  2.,  3.,
    4.,  5.,  6.,  7.,
    8.,  9.,  10., 11.,
    0.,  1.,  2.,  3.,
    4.,  5.,  6.,  7.,

    12., 13., 14., 15.,
    16., 17., 18., 19.,
    20., 21., 22., 23.,
    8.,  9.,  10., 11.,
    12., 13., 14., 15.
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}
