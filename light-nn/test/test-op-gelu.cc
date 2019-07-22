#include "operators/gelu.h"

#include "gtest/gtest.h"

class GELUTest : public testing::Test {
 protected:
  virtual void SetUp() {
    // construct json config
    m_config["name"] = "gelu1";

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

TEST_F(GELUTest, Forward) {
  lnn::GELU GELU(m_config);

  float input_data[] = {
    0.5, -0.3, 0.1,  0.9,
    0.0, -0.8, -0.5, 1.4
  };
  std::vector<size_t> shape(2);
  shape[0] = 2;
  shape[1] = 4;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(GELU.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(8, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(2, m_output_vec[0]->shape(0));
  EXPECT_EQ(4, m_output_vec[0]->shape(1));
  float target[] = {
    0.345731, -0.114626, 0.053983, 0.734346,
    0.0,      -0.169484, -0.154269, 1.286941
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}
