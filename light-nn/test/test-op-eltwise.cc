#include "operators/eltwise.h"

#include "gtest/gtest.h"

class EltwiseTest : public testing::Test {
 protected:
  virtual void SetUp() {
    // construct json config
    m_config["name"] = "eltwise";
    Json::Value param;
    param["type"] = "SUM";
    m_config["param"] = param;

    // construct input & output tensor vector
    m_input_tensor1 = new lnn::Tensor;
    m_input_tensor2 = new lnn::Tensor;
    m_output_tensor = new lnn::Tensor;
    m_input_vec.push_back(m_input_tensor1);
    m_input_vec.push_back(m_input_tensor2);
    m_output_vec.push_back(m_output_tensor);
  }
  virtual void TearDown() {
    delete m_input_tensor1;
    delete m_input_tensor2;
    delete m_output_tensor;
  }

  Json::Value m_config;
  lnn::Tensor *m_input_tensor1;
  lnn::Tensor *m_input_tensor2;
  lnn::Tensor *m_output_tensor;
  std::vector<lnn::Tensor *> m_input_vec;
  std::vector<lnn::Tensor *> m_output_vec;
};

TEST_F(EltwiseTest, TypePROD) {
  m_config["param"]["type"] = "PROD";
  lnn::Eltwise eltwise(m_config);

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 3;
  shape[1] = 5;
  m_input_tensor1->set_data(input_data, shape);
  m_input_tensor2->set_data(input_data, shape);

  EXPECT_TRUE(eltwise.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(15, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(3, m_output_vec[0]->shape(0));
  EXPECT_EQ(5, m_output_vec[0]->shape(1));
  float target[] = {
    0., 1., 4., 9., 16.,
    25., 36., 49., 64., 81.,
    100., 121., 144., 169., 196.,
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(EltwiseTest, TypeSUM) {
  m_config["param"]["type"] = "SUM";
  lnn::Eltwise eltwise(m_config);

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 3;
  shape[1] = 5;
  m_input_tensor1->set_data(input_data, shape);
  m_input_tensor2->set_data(input_data, shape);

  EXPECT_TRUE(eltwise.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(15, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(3, m_output_vec[0]->shape(0));
  EXPECT_EQ(5, m_output_vec[0]->shape(1));
  float target[] = {
    0., 2., 4., 6., 8.,
    10., 12., 14., 16., 18.,
    20., 22., 24., 26., 28.
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(EltwiseTest, TypeMAX) {
  m_config["param"]["type"] = "MAX";
  lnn::Eltwise eltwise(m_config);

  float input_data1[15], input_data2[15];
  for (int i = 0; i < 15; ++i) {
    input_data1[i] = i;
    input_data2[i] = 14 - i;
  }
  std::vector<size_t> shape(2);
  shape[0] = 3;
  shape[1] = 5;
  m_input_tensor1->set_data(input_data1, shape);
  m_input_tensor2->set_data(input_data2, shape);

  EXPECT_TRUE(eltwise.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(15, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(3, m_output_vec[0]->shape(0));
  EXPECT_EQ(5, m_output_vec[0]->shape(1));
  float target[] = {
    14., 13., 12., 11., 10.,
    9., 8., 7., 8., 9.,
    10., 11., 12., 13., 14.
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}
