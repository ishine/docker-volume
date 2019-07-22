#include "operators/scale.h"

#include "gtest/gtest.h"

class ScaleTest : public testing::Test {
 protected:
  virtual void SetUp() {
    // construct json config
    m_config["name"] = "scale1";
    Json::Value param;
    m_config["param"] = param;

    // construct input & output tensor vector
    m_input_tensor = new lnn::Tensor;
    m_input_tensor_other = new lnn::Tensor;
    m_output_tensor = new lnn::Tensor;
    m_input_vec.push_back(m_input_tensor);
    m_output_vec.push_back(m_output_tensor);
  }
  virtual void TearDown() {
    delete m_input_tensor;
    delete m_input_tensor_other;
    delete m_output_tensor;
  }

  Json::Value m_config;
  lnn::Tensor *m_input_tensor;
  lnn::Tensor *m_input_tensor_other;
  lnn::Tensor *m_output_tensor;
  std::vector<lnn::Tensor *> m_input_vec;
  std::vector<lnn::Tensor *> m_output_vec;
};

TEST_F(ScaleTest, OneInput) {
  m_config["param"]["scale"] = 0.5;
  lnn::Scale scale(m_config);

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 3;
  shape[1] = 5;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(scale.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(15, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(3, m_output_vec[0]->shape(0));
  EXPECT_EQ(5, m_output_vec[0]->shape(1));
  float target[] = {
    0., 0.5, 1., 1.5, 2.,
    2.5, 3., 3.5, 4., 4.5,
    5., 5.5, 6., 6.5, 7.
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(ScaleTest, TwoInput) {
  lnn::Scale scale(m_config);

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 3;
  shape[1] = 5;
  m_input_tensor->set_data(input_data, shape);
  float scale_data[] = { 0.2, 0.5, 1. };
  shape.resize(1);
  m_input_tensor_other->set_data(scale_data, shape);

  m_input_vec.push_back(m_input_tensor_other);

  EXPECT_TRUE(scale.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(15, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(3, m_output_vec[0]->shape(0));
  EXPECT_EQ(5, m_output_vec[0]->shape(1));
  float target[] = {
    0., 0.2, 0.4, 0.6, 0.8,
    2.5, 3., 3.5, 4., 4.5,
    10., 11., 12., 13., 14.
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}
