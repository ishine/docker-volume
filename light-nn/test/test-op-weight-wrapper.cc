#include "operators/weight-wrapper.h"

#include "gtest/gtest.h"

class WeightWrapperTest : public testing::Test {
 protected:
  virtual void SetUp() {
    // construct json config
    int input_size(3), output_size(5);
    m_config["name"] = "weight-wrapper1";
    Json::Value param;
    param["input_size"] = input_size;
    param["output_size"] = output_size;
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
    m_weights[0].set_name("weight");
    m_weights[0].set_data(m_weight_data, shape);
    shape.resize(1);
    m_weights[1].set_name("bias");
    m_weights[1].set_data(m_bias_data, shape);
    m_name2id.clear();
    m_name2id.insert(std::make_pair("weight", 0));
    m_name2id.insert(std::make_pair("bias", 1));

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

TEST_F(WeightWrapperTest, 1D) {
  m_config["param"]["input_size"] = 1;
  m_config["param"]["weight_name"] = "bias";
  lnn::WeightWrapper weight_wrapper(m_config);
  EXPECT_TRUE(weight_wrapper.set_weight(m_weights, m_name2id));

  EXPECT_TRUE(weight_wrapper.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(5, m_output_vec[0]->size());
  EXPECT_EQ(1, m_output_vec[0]->num_axes());
  EXPECT_EQ(5, m_output_vec[0]->shape(0));
  float target[] = {
    0., 0.1, 0.2, 0.3, 0.4
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(WeightWrapperTest, 2D) {
  m_config["param"]["weight_name"] = "weight";
  lnn::WeightWrapper weight_wrapper(m_config);
  EXPECT_TRUE(weight_wrapper.set_weight(m_weights, m_name2id));

  EXPECT_TRUE(weight_wrapper.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(15, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(5, m_output_vec[0]->shape(0));
  EXPECT_EQ(3, m_output_vec[0]->shape(1));
  float target[] = {
    0., 1., 2.,
    3., 4., 5.,
    6., 7., 8.,
    9., 10., 11.,
    12., 13., 14.
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}
