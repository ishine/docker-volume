#include "operators/softmax.h"

#include "gtest/gtest.h"

class SoftmaxTest : public testing::Test {
 protected:
  virtual void SetUp() {
    // construct json config
    m_config["name"] = "softmax1";

    // construct input & output tensor vector
    m_input_tensor = new lnn::Tensor;
    m_output_tensor = new lnn::Tensor;
    m_input_vec.push_back(m_input_tensor);
    m_output_vec.push_back(m_output_tensor);
  }
  virtual void TearDown() {
    delete m_input_tensor;
    delete m_output_tensor;
  }

  Json::Value m_config;
  lnn::Tensor *m_input_tensor;
  lnn::Tensor *m_output_tensor;
  std::vector<lnn::Tensor *> m_input_vec;
  std::vector<lnn::Tensor *> m_output_vec;
};

TEST_F(SoftmaxTest, ForwardAxisDefault) {
  lnn::Softmax softmax(m_config);

  float input_data[] = {
    0.5, -0.3, 0.1,  0.9,
    0.0, -0.8, -0.5, 1.4
  };
  std::vector<size_t> shape(2);
  shape[0] = 2;
  shape[1] = 4;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(softmax.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(8, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(2, m_output_vec[0]->shape(0));
  EXPECT_EQ(4, m_output_vec[0]->shape(1));
  float target[] = {
    0.276895, 0.124417, 0.185608, 0.413079,
    0.163638, 0.073527, 0.099251, 0.663584
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(SoftmaxTest, ForwardAxisSet) {
  Json::Value param;
  param["axis"] = 0;
  m_config["param"] = param;
  lnn::Softmax softmax(m_config);

  float input_data[] = {
    0.5, -0.3, 0.1,  0.9,
    0.0, -0.8, -0.5, 1.4
  };
  std::vector<size_t> shape(2);
  shape[0] = 2;
  shape[1] = 4;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(softmax.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(8, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(2, m_output_vec[0]->shape(0));
  EXPECT_EQ(4, m_output_vec[0]->shape(1));
  float target[] = {
    0.136649, 0.061400, 0.091599, 0.203856,
    0.082882, 0.037241, 0.050270, 0.336102
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}
