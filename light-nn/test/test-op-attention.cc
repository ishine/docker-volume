#include "operators/attention.h"

#include "gtest/gtest.h"

class AttentionTest : public testing::Test {
 protected:
  virtual void SetUp() {
    // construct json config
    m_config["name"] = "attention1";

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

TEST_F(AttentionTest, Basic) {
  lnn::Attention attention(m_config);

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 3;
  shape[1] = 5;
  m_input_tensor->set_data(input_data, shape);
  float input_data2[] = { 0.05, 0.1, 0.15, 0.2, 0.25 };
  shape.resize(1);
  shape[0] = 5;
  m_input_tensor_2->set_data(input_data2, shape);

  EXPECT_TRUE(attention.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(5, m_output_vec[0]->size());
  EXPECT_EQ(1, m_output_vec[0]->num_axes());
  EXPECT_EQ(5, m_output_vec[0]->shape(0));
  float target[] = {
    9.879774, 10.879773, 11.879774, 12.879774, 13.879774
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}
