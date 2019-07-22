#include "operators/linear.h"

#include "gtest/gtest.h"

class LinearTest : public testing::Test {
 protected:
  virtual void SetUp() {
    // construct json config
    int input_size(5), output_size(10);
    m_config["name"] = "linear1";
    Json::Value param;
    param["input_size"] = input_size;
    param["output_size"] = output_size;
    param["bias"] = true;
    m_config["param"] = param;

    // construct weight & bias tensor
    m_weight_data = new float[output_size*input_size];
    m_bias_data = new float[output_size];
    for (int i = 0; i < output_size*input_size; ++i) {
      m_weight_data[i] = i;
    }
    for (int i = 0; i < output_size; ++i) {
      m_bias_data[i] = i / 10.;
    }
    std::vector<size_t> shape(2);
    shape[0] = output_size;
    shape[1] = input_size;
    m_weights.resize(2);
    m_weights[0].set_name("linear1.weight");
    m_weights[0].set_data(m_weight_data, shape);
    shape.resize(1);
    m_weights[1].set_name("linear1.bias");
    m_weights[1].set_data(m_bias_data, shape);
    m_name2id.clear();
    m_name2id.insert(std::make_pair("linear1.weight", 0));
    m_name2id.insert(std::make_pair("linear1.bias", 1));

    // construct input & output tensor vector
    m_input_tensor = new lnn::Tensor;
    m_output_tensor = new lnn::Tensor;
    m_input_vec.push_back(m_input_tensor);
    m_output_vec.push_back(m_output_tensor);
  }
  virtual void TearDown() {
    delete [] m_weight_data;
    delete [] m_bias_data;
    delete m_input_tensor;
    delete m_output_tensor;
  }

  Json::Value m_config;
  float *m_weight_data, *m_bias_data;
  std::vector<lnn::Tensor> m_weights;
  std::map<std::string, size_t> m_name2id;
  lnn::Tensor *m_input_tensor;
  lnn::Tensor *m_output_tensor;
  std::vector<lnn::Tensor *> m_input_vec;
  std::vector<lnn::Tensor *> m_output_vec;
};

TEST_F(LinearTest, WithBias) {
  lnn::Linear linear(m_config);
  EXPECT_TRUE(linear.set_weight(m_weights, m_name2id));

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 3;
  shape[1] = 5;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(linear.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(30, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(3, m_output_vec[0]->shape(0));
  EXPECT_EQ(10, m_output_vec[0]->shape(1));
  float target[] = {
    30.0,  80.1,  130.2, 180.3,  230.4,  280.5,  330.6,  380.7,  430.8,  480.9,
    80.0,  255.1, 430.2, 605.3,  780.4,  955.5,  1130.6, 1305.7, 1480.8, 1655.9,
    130.0, 430.1, 730.2, 1030.3, 1330.4, 1630.5, 1930.6, 2230.7, 2530.8, 2830.9
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(LinearTest, WithoutBias) {
  m_config["param"]["bias"] = false;
  lnn::Linear linear(m_config);
  EXPECT_TRUE(linear.set_weight(m_weights, m_name2id));

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 3;
  shape[1] = 5;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(linear.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(30, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(3, m_output_vec[0]->shape(0));
  EXPECT_EQ(10, m_output_vec[0]->shape(1));
  float target[] = {
    30.,  80.,  130., 180.,  230.,  280.,  330.,  380.,  430.,  480.,
    80.,  255., 430., 605.,  780.,  955.,  1130., 1305., 1480., 1655.,
    130., 430., 730., 1030., 1330., 1630., 1930., 2230., 2530., 2830.
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}
