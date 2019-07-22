#include "operators/lstm.h"

#include "gtest/gtest.h"

class LSTMTest : public testing::Test {
 protected:
  virtual void SetUp() {
    // construct json config
    int input_size(3), output_size(5);
    m_config["name"] = "lstm1";
    Json::Value param;
    param["input_size"] = input_size;
    param["output_size"] = output_size;
    param["bias"] = true;
    param["bidirectional"] = true;
    m_config["param"] = param;

    // construct weight & bias tensor
    size_t s1 = 4*output_size*input_size;
    size_t s2 = 4*output_size*output_size;
    size_t s3 = 4*output_size;
    m_xh_data = new float[s1];
    m_hh_data = new float[s2];
    m_bias_data = new float[s3];
    m_xh_data_rev = new float[s1];
    m_hh_data_rev = new float[s2];
    m_bias_data_rev = new float[s3];
    for (size_t i = 0; i < s1; ++i) {
      m_xh_data[i] = i * 1.0 / s1;
      m_xh_data_rev[i] = (s1 - i) * 1.0 / s1;
    }
    for (size_t i = 0; i < s2; ++i) {
      m_hh_data[i] = (i + 1) * 1.0 / s2;
      m_hh_data_rev[i] = (s2 - i - 1) * 1.0 / s2;
    }
    for (size_t i = 0; i < s3; ++i) {
      m_bias_data[i] = i * 1.0 / s3;
      m_bias_data_rev[i] = i * 1.0 / s3;
    }
    std::vector<size_t> shape(2);
    shape[0] = 4*output_size;
    shape[1] = input_size;
    m_weights.resize(6);
    m_weights[0].set_name("lstm1.w_ih");
    m_weights[0].set_data(m_xh_data, shape);
    m_weights[3].set_name("lstm1.w_ih_reverse");
    m_weights[3].set_data(m_xh_data_rev, shape);
    shape[1] = output_size;
    m_weights[1].set_name("lstm1.w_hh");
    m_weights[1].set_data(m_hh_data, shape);
    m_weights[4].set_name("lstm1.w_hh_reverse");
    m_weights[4].set_data(m_hh_data_rev, shape);
    shape.resize(1);
    m_weights[2].set_name("lstm1.b");
    m_weights[2].set_data(m_bias_data, shape);
    m_weights[5].set_name("lstm1.b_reverse");
    m_weights[5].set_data(m_bias_data_rev, shape);
    m_name2id.clear();
    m_name2id.insert(std::make_pair("lstm1.w_ih", 0));
    m_name2id.insert(std::make_pair("lstm1.w_hh", 1));
    m_name2id.insert(std::make_pair("lstm1.b", 2));
    m_name2id.insert(std::make_pair("lstm1.w_ih_reverse", 3));
    m_name2id.insert(std::make_pair("lstm1.w_hh_reverse", 4));
    m_name2id.insert(std::make_pair("lstm1.b_reverse", 5));

    // construct input & output tensor vector
    m_input_tensor = new lnn::Tensor;
    m_output_tensor = new lnn::Tensor;
    m_input_vec.push_back(m_input_tensor);
    m_output_vec.push_back(m_output_tensor);
  }
  virtual void TearDown() {
    delete [] m_xh_data;
    delete [] m_hh_data;
    delete [] m_bias_data;
    delete [] m_xh_data_rev;
    delete [] m_hh_data_rev;
    delete [] m_bias_data_rev;
    delete m_input_tensor;
    delete m_output_tensor;
  }

  Json::Value m_config;
  float *m_xh_data, *m_hh_data, *m_bias_data;
  float *m_xh_data_rev, *m_hh_data_rev, *m_bias_data_rev;
  std::vector<lnn::Tensor> m_weights;
  std::map<std::string, size_t> m_name2id;
  lnn::Tensor *m_input_tensor;
  lnn::Tensor *m_output_tensor;
  std::vector<lnn::Tensor *> m_input_vec;
  std::vector<lnn::Tensor *> m_output_vec;
};

TEST_F(LSTMTest, BidirectionalFalseBiasFalse) {
  m_config["param"]["bidirectional"] = false;
  m_config["param"]["bias"] = false;
  lnn::LSTM lstm(m_config);
  EXPECT_TRUE(lstm.set_weight(m_weights, m_name2id));

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 5;
  shape[1] = 3;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(lstm.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(25, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(5, m_output_vec[0]->shape(0));
  EXPECT_EQ(5, m_output_vec[0]->shape(1));
  float target[] = {
    0.390624, 0.425583, 0.459053, 0.490687, 0.520236,
    0.791417, 0.856159, 0.892675, 0.912788, 0.924497,
    0.935654, 0.972467, 0.983363, 0.98742,  0.989363,
    0.982526, 0.995564, 0.997662, 0.998279, 0.998552,
    0.995636, 0.999341, 0.99968,  0.999767, 0.999804
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(LSTMTest, BidirectionalFalseBiasTrue) {
  m_config["param"]["bidirectional"] = false;
  m_config["param"]["bias"] = true;
  lnn::LSTM lstm(m_config);
  EXPECT_TRUE(lstm.set_weight(m_weights, m_name2id));

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 5;
  shape[1] = 3;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(lstm.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(25, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(5, m_output_vec[0]->shape(0));
  EXPECT_EQ(5, m_output_vec[0]->shape(1));
  float target[] = {
    0.423872, 0.466856, 0.506979, 0.543633, 0.576484,
    0.796067, 0.864895, 0.901636, 0.921048, 0.93215,
    0.937198, 0.974545, 0.984934, 0.988692, 0.990485,
    0.982958, 0.995932, 0.99789,  0.998455, 0.998706,
    0.995745, 0.999398, 0.999711, 0.999791, 0.999825
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(LSTMTest, BidirectionalTrueBiasFalse) {
  m_config["param"]["bidirectional"] = true;
  m_config["param"]["bias"] = false;
  lnn::LSTM lstm(m_config);
  EXPECT_TRUE(lstm.set_weight(m_weights, m_name2id));

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 5;
  shape[1] = 3;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(lstm.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(50, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(5, m_output_vec[0]->shape(0));
  EXPECT_EQ(10, m_output_vec[0]->shape(1));
  float target[] = {
    0.390624, 0.425583, 0.459053, 0.490687, 0.520236, 0.977015, 0.96617,  0.95043,  0.927783, 0.890374,
    0.791417, 0.856159, 0.892675, 0.912788, 0.924497, 0.999024, 0.998613, 0.997634, 0.995185, 0.982293,
    0.935654, 0.972467, 0.983363, 0.98742,  0.989363, 0.99505,  0.995037, 0.994971, 0.994397, 0.978352,
    0.982526, 0.995564, 0.997662, 0.998279, 0.998552, 0.964027, 0.964026, 0.964,    0.963179, 0.926259,
    0.995636, 0.999341, 0.99968,  0.999767, 0.999804, 0.761594, 0.761594, 0.761567, 0.760241, 0.692566
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(LSTMTest, BidirectionalTrueBiasTrue) {
  m_config["param"]["bidirectional"] = true;
  m_config["param"]["bias"] = true;
  lnn::LSTM lstm(m_config);
  EXPECT_TRUE(lstm.set_weight(m_weights, m_name2id));

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 5;
  shape[1] = 3;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(lstm.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(50, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(5, m_output_vec[0]->shape(0));
  EXPECT_EQ(10, m_output_vec[0]->shape(1));
  float target[] = {
    0.423872, 0.466856, 0.506979, 0.543633, 0.576484, 0.985998, 0.980292, 0.97231,  0.961201, 0.945799,
    0.796067, 0.864895, 0.901636, 0.921048, 0.93215,  0.999146, 0.998921, 0.998418, 0.997279, 0.994479,
    0.937198, 0.974545, 0.984934, 0.988692, 0.990485, 0.995052, 0.995046, 0.995019, 0.994877, 0.993319,
    0.982958, 0.995932, 0.99789,  0.998455, 0.998706, 0.964028, 0.964027, 0.964022, 0.963884, 0.959502,
    0.995745, 0.999398, 0.999711, 0.999791, 0.999825, 0.761594, 0.761594, 0.761589, 0.76137,  0.751553
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}
