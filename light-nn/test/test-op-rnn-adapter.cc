#include "operators/rnn-adapter.h"

#include "gtest/gtest.h"

class RNNAdapterTest : public testing::Test {
 protected:
  virtual void SetUp() {
    // construct json config
    m_config["name"] = "adapter1";
    Json::Value param;
    param["type"] = "LAST";
    m_config["param"] = param;

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

TEST_F(RNNAdapterTest, TypeLast) {
  m_config["param"]["type"] = "LAST";
  lnn::RNNAdapter adapter(m_config);

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 3;
  shape[1] = 5;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(adapter.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(5, m_output_vec[0]->size());
  EXPECT_EQ(1, m_output_vec[0]->num_axes());
  EXPECT_EQ(5, m_output_vec[0]->shape(0));
  float target[] = { 10., 11., 12., 13., 14. };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(RNNAdapterTest, TypeAverage) {
  m_config["param"]["type"] = "AVERAGE";
  lnn::RNNAdapter adapter(m_config);

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 3;
  shape[1] = 5;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(adapter.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(5, m_output_vec[0]->size());
  EXPECT_EQ(1, m_output_vec[0]->num_axes());
  EXPECT_EQ(5, m_output_vec[0]->shape(0));
  float target[] = { 5., 6., 7., 8., 9. };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}
