#include "operators/weighted-sum.h"

#include "gtest/gtest.h"

class WeightedSumTest : public testing::Test {
 protected:
  virtual void SetUp() {
    // construct json config
    m_config["name"] = "weighted-sum";

    // construct input & output tensor vector
    m_input_tensor1 = new lnn::Tensor;
    m_input_tensor2 = new lnn::Tensor;
    m_input_tensor3 = new lnn::Tensor;
    m_output_tensor = new lnn::Tensor;
    m_input_vec.push_back(m_input_tensor1);
    m_input_vec.push_back(m_input_tensor2);
    m_input_vec.push_back(m_input_tensor3);
    m_output_vec.push_back(m_output_tensor);
  }
  virtual void TearDown() {
    delete m_input_tensor1;
    delete m_input_tensor2;
    delete m_input_tensor3;
    delete m_output_tensor;
  }

  Json::Value m_config;
  lnn::Tensor *m_input_tensor1;
  lnn::Tensor *m_input_tensor2;
  lnn::Tensor *m_input_tensor3;
  lnn::Tensor *m_output_tensor;
  std::vector<lnn::Tensor *> m_input_vec;
  std::vector<lnn::Tensor *> m_output_vec;
};

TEST_F(WeightedSumTest, Scalar) {
  lnn::WeightedSum weighted_sum(m_config);

  float input_data1[15], input_data2[15];
  float lambda = 0.5;
  for (int i = 0; i < 15; ++i) {
    input_data1[i] = i;
    input_data2[i] = 14 - i;
  }
  std::vector<size_t> shape(2);
  shape[0] = 3;
  shape[1] = 5;
  m_input_tensor1->set_data(input_data1, shape);
  m_input_tensor2->set_data(input_data2, shape);
  shape.resize(1);
  shape[0] = 1;
  m_input_tensor3->set_data(&lambda, shape);

  EXPECT_TRUE(weighted_sum.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(15, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(3, m_output_vec[0]->shape(0));
  EXPECT_EQ(5, m_output_vec[0]->shape(1));
  float target[] = {
    7., 7., 7., 7., 7.,
    7., 7., 7., 7., 7.,
    7., 7., 7., 7., 7.
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(WeightedSumTest, Tensor) {
  lnn::WeightedSum weighted_sum(m_config);

  float input_data1[15], input_data2[15], input_data3[15];
  for (int i = 0; i < 15; ++i) {
    input_data1[i] = i;
    input_data2[i] = 14 - i;
    input_data3[i] = 0.5;
  }
  std::vector<size_t> shape(2);
  shape[0] = 3;
  shape[1] = 5;
  m_input_tensor1->set_data(input_data1, shape);
  m_input_tensor2->set_data(input_data2, shape);
  m_input_tensor3->set_data(input_data3, shape);

  EXPECT_TRUE(weighted_sum.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(15, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(3, m_output_vec[0]->shape(0));
  EXPECT_EQ(5, m_output_vec[0]->shape(1));
  float target[] = {
    7., 7., 7., 7., 7.,
    7., 7., 7., 7., 7.,
    7., 7., 7., 7., 7.
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}
