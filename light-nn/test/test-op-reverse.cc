#include "operators/reverse.h"

#include "gtest/gtest.h"

class ReverseTest : public testing::Test {
 protected:
  virtual void SetUp() {
    // construct json config
    m_config["name"] = "reverse1";

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

TEST_F(ReverseTest, TypeLast) {
  lnn::Reverse reverse(m_config);

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 5;
  shape[1] = 3;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(reverse.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(15, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(5, m_output_vec[0]->shape(0));
  EXPECT_EQ(3, m_output_vec[0]->shape(1));
  float target[] = {
    12., 13., 14.,
    9., 10., 11.,
    6., 7., 8.,
    3., 4., 5.,
    0., 1., 2.,
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}
