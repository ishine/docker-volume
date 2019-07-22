#include "operators/conv1d.h"

#include "gtest/gtest.h"

class Conv1DTest : public testing::Test {
 protected:
  virtual void SetUp() {
    // construct json config
    int input_size(3), output_size(5), kernel_size(3);
    m_config["name"] = "conv1d";
    Json::Value param;
    param["input_size"] = input_size;
    param["output_size"] = output_size;
    param["kernel_size"] = kernel_size;
    param["bias"] = true;
    m_config["param"] = param;

    // construct weight & bias tensor
    size_t s1 = output_size*kernel_size*input_size;
    size_t s2 = output_size;
    m_weight_data = new float[s1];
    m_bias_data = new float[s2];
    for (size_t i = 0; i < s1; ++i) {
      m_weight_data[i] = i * 1.0 / s1;
    }
    for (size_t i = 0; i < s2; ++i) {
      m_bias_data[i] = i * 1.0 / s2;
    }
    std::vector<size_t> shape(3);
    shape[0] = output_size;
    shape[1] = kernel_size;
    shape[2] = input_size;
    m_weights.resize(2);
    m_weights[0].set_name("conv1d.weight");
    m_weights[0].set_data(m_weight_data, shape);
    shape.resize(1);
    m_weights[1].set_name("conv1d.bias");
    m_weights[1].set_data(m_bias_data, shape);
    m_name2id.clear();
    m_name2id.insert(std::make_pair("conv1d.weight", 0));
    m_name2id.insert(std::make_pair("conv1d.bias", 1));

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

TEST_F(Conv1DTest, Default) {
  lnn::Conv1D conv1d(m_config);
  EXPECT_TRUE(conv1d.set_weight(m_weights, m_name2id));

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 5;
  shape[1] = 3;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(conv1d.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(15, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(3, m_output_vec[0]->shape(0));
  EXPECT_EQ(5, m_output_vec[0]->shape(1));
  float target[] = {
    4.533333, 11.933333, 19.333334, 26.733334, 34.133335,
    6.933333, 19.733334, 32.533333, 45.333336, 58.133331,
    9.333333, 27.533335, 45.733337, 63.933334, 82.133331
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(Conv1DTest, Padding) {
  m_config["param"]["padding"] = 1;
  lnn::Conv1D conv1d(m_config);
  EXPECT_TRUE(conv1d.set_weight(m_weights, m_name2id));

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 5;
  shape[1] = 3;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(conv1d.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(25, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(5, m_output_vec[0]->shape(0));
  EXPECT_EQ(5, m_output_vec[0]->shape(1));
  float target[] = {
    2.222222, 5.422222, 8.622223, 11.822222, 15.022223,
    4.533333, 11.933333, 19.333334, 26.733334, 34.133335,
    6.933333, 19.733334, 32.533333, 45.333336, 58.133331,
    9.333333, 27.533335, 45.733337, 63.933334, 82.133331,
    4.222222, 18.222223, 32.222225, 46.222221, 60.222225
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(Conv1DTest, Stride) {
  m_config["param"]["stride"] = 2;
  lnn::Conv1D conv1d(m_config);
  EXPECT_TRUE(conv1d.set_weight(m_weights, m_name2id));

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 5;
  shape[1] = 3;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(conv1d.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(10, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(2, m_output_vec[0]->shape(0));
  EXPECT_EQ(5, m_output_vec[0]->shape(1));
  float target[] = {
    4.533333, 11.933333, 19.333334, 26.733334, 34.133335,
    9.333333, 27.533335, 45.733337, 63.933334, 82.133331
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(Conv1DTest, Dilation) {
  m_config["param"]["dilation"] = 2;
  lnn::Conv1D conv1d(m_config);
  EXPECT_TRUE(conv1d.set_weight(m_weights, m_name2id));

  float input_data[18];
  for (int i = 0; i < 18; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 6;
  shape[1] = 3;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(conv1d.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(10, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(2, m_output_vec[0]->shape(0));
  EXPECT_EQ(5, m_output_vec[0]->shape(1));
  float target[] = {
    8.133334, 20.933334, 33.733334, 46.533333, 59.333332,
    10.533334, 28.733335, 46.933338, 65.133331, 83.333336
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}
