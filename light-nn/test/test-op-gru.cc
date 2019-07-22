#include "operators/gru.h"

#include "gtest/gtest.h"

class GRUTest : public testing::Test {
 protected:
  virtual void SetUp() {
    // construct json config
    int input_size(3), output_size(5);
    m_config["name"] = "gru1";
    Json::Value param;
    param["input_size"] = input_size;
    param["output_size"] = output_size;
    param["bias"] = true;
    param["bidirectional"] = true;
    m_config["param"] = param;

    // construct weight & bias tensor
    size_t s1 = 3*output_size*input_size;
    size_t s2 = 3*output_size*output_size;
    size_t s3 = 3*output_size;
    m_xh_data = new float[s1];
    m_hh_data = new float[s2];
    m_b_ih_data = new float[s3];
    m_b_hh_data = new float[s3];
    m_xh_data_rev = new float[s1];
    m_hh_data_rev = new float[s2];
    m_b_ih_data_rev = new float[s3];
    m_b_hh_data_rev = new float[s3];
    for (size_t i = 0; i < s1; ++i) {
      m_xh_data[i] = i * 1.0 / s1;
      m_xh_data_rev[i] = (s1 - i) * 1.0 / s1;
    }
    for (size_t i = 0; i < s2; ++i) {
      m_hh_data[i] = (i + 1) * 1.0 / s2;
      m_hh_data_rev[i] = (s2 - i - 1) * 1.0 / s2;
    }
    for (size_t i = 0; i < s3; ++i) {
      m_b_ih_data[i] = i * 1.0 / s3;
      m_b_hh_data[i] = i * 1.0 / s3;
      m_b_ih_data_rev[i] = i * 1.0 / s3;
      m_b_hh_data_rev[i] = i * 1.0 / s3;
    }
    std::vector<size_t> shape(2);
    shape[0] = 3*output_size;
    shape[1] = input_size;
    m_weights.resize(8);
    m_weights[0].set_name("gru1.w_ih");
    m_weights[0].set_data(m_xh_data, shape);
    m_weights[4].set_name("gru1.w_ih_reverse");
    m_weights[4].set_data(m_xh_data_rev, shape);
    shape[1] = output_size;
    m_weights[1].set_name("gru1.w_hh");
    m_weights[1].set_data(m_hh_data, shape);
    m_weights[5].set_name("gru1.w_hh_reverse");
    m_weights[5].set_data(m_hh_data_rev, shape);
    shape.resize(1);
    m_weights[2].set_name("gru1.b_ih");
    m_weights[2].set_data(m_b_ih_data, shape);
    m_weights[3].set_name("gru1.b_ih");
    m_weights[3].set_data(m_b_hh_data, shape);
    m_weights[6].set_name("gru1.b_ih_reverse");
    m_weights[6].set_data(m_b_ih_data_rev, shape);
    m_weights[7].set_name("gru1.b_ih_reverse");
    m_weights[7].set_data(m_b_hh_data_rev, shape);
    m_name2id.clear();
    m_name2id.insert(std::make_pair("gru1.w_ih", 0));
    m_name2id.insert(std::make_pair("gru1.w_hh", 1));
    m_name2id.insert(std::make_pair("gru1.b_ih", 2));
    m_name2id.insert(std::make_pair("gru1.b_hh", 3));
    m_name2id.insert(std::make_pair("gru1.w_ih_reverse", 4));
    m_name2id.insert(std::make_pair("gru1.w_hh_reverse", 5));
    m_name2id.insert(std::make_pair("gru1.b_ih_reverse", 6));
    m_name2id.insert(std::make_pair("gru1.b_hh_reverse", 7));

    // construct input & output tensor vector
    m_input_tensor = new lnn::Tensor;
    m_output_tensor = new lnn::Tensor;
    m_input_vec.push_back(m_input_tensor);
    m_output_vec.push_back(m_output_tensor);
  }
  virtual void TearDown() {
    delete [] m_xh_data;
    delete [] m_hh_data;
    delete [] m_b_ih_data;
    delete [] m_b_hh_data;
    delete [] m_xh_data_rev;
    delete [] m_hh_data_rev;
    delete [] m_b_ih_data_rev;
    delete [] m_b_hh_data_rev;
    delete m_input_tensor;
    delete m_output_tensor;
  }

  Json::Value m_config;
  float *m_xh_data, *m_hh_data, *m_b_ih_data, *m_b_hh_data;
  float *m_xh_data_rev, *m_hh_data_rev, *m_b_ih_data_rev, *m_b_hh_data_rev;
  std::vector<lnn::Tensor> m_weights;
  std::map<std::string, size_t> m_name2id;
  lnn::Tensor *m_input_tensor;
  lnn::Tensor *m_output_tensor;
  std::vector<lnn::Tensor *> m_input_vec;
  std::vector<lnn::Tensor *> m_output_vec;
};

TEST_F(GRUTest, BidirectionalFalseBiasFalse) {
  m_config["param"]["bidirectional"] = false;
  m_config["param"]["bias"] = false;
  lnn::GRU gru(m_config);
  EXPECT_TRUE(gru.set_weight(m_weights, m_name2id));

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 5;
  shape[1] = 3;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(gru.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(25, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(5, m_output_vec[0]->shape(0));
  EXPECT_EQ(5, m_output_vec[0]->shape(1));
  float target[] = {
    0.240504, 0.208167, 0.178407, 0.151674, 0.128095,
    0.247727, 0.21137,  0.179816, 0.15229,  0.128363,
    0.24802,  0.211441, 0.179833, 0.152294, 0.128364,
    0.248032, 0.211443, 0.179833, 0.152294, 0.128364,
    0.248032, 0.211443, 0.179833, 0.152294, 0.128364
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(GRUTest, BidirectionalFalseBiasTrue) {
  m_config["param"]["bidirectional"] = false;
  m_config["param"]["bias"] = true;
  lnn::GRU gru(m_config);
  EXPECT_TRUE(gru.set_weight(m_weights, m_name2id));

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 5;
  shape[1] = 3;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(gru.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(25, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(5, m_output_vec[0]->shape(0));
  EXPECT_EQ(5, m_output_vec[0]->shape(1));
  float target[] = {
    0.144026, 0.107821, 0.079775, 0.058512, 0.042643,
    0.14903,  0.109821, 0.080564, 0.058821, 0.042763,
    0.149234, 0.109866, 0.080574, 0.058823, 0.042763,
    0.149242, 0.109867, 0.080574, 0.058823, 0.042763,
    0.149242, 0.109867, 0.080574, 0.058823, 0.042763
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(GRUTest, BidirectionalTrueBiasFalse) {
  m_config["param"]["bidirectional"] = true;
  m_config["param"]["bias"] = false;
  lnn::GRU gru(m_config);
  EXPECT_TRUE(gru.set_weight(m_weights, m_name2id));

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 5;
  shape[1] = 3;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(gru.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(50, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(5, m_output_vec[0]->shape(0));
  EXPECT_EQ(10, m_output_vec[0]->shape(1));
  float target[] = {
    0.240504, 0.208167, 0.178407, 0.151674, 0.128095, 0.093347, 0.093679, 0.085092, 0.064119, 0.026266,
    0.247727, 0.21137,  0.179816, 0.15229,  0.128363, 0.000458, 0.001017, 0.002217, 0.004404, 0.005310,
    0.24802,  0.211441, 0.179833, 0.152294, 0.128364, 1.39047E-6, 5.65231E-6, 0.000023, 0.000092, 0.000278,
    0.248032, 0.211443, 0.179833, 0.152294, 0.128364, 4.20973E-9, 3.11831E-8, 2.31442E-7, 1.7195E-6, 0.000011,
    0.248032, 0.211443, 0.179833, 0.152294, 0.128364, 1.27067E-11, 1.7108E-10, 2.30337E-9, 3.10004E-8, 3.89983E-7
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(GRUTest, BidirectionalTrueBiasTrue) {
  m_config["param"]["bidirectional"] = true;
  m_config["param"]["bias"] = true;
  lnn::GRU gru(m_config);
  EXPECT_TRUE(gru.set_weight(m_weights, m_name2id));

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 5;
  shape[1] = 3;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(gru.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(50, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(5, m_output_vec[0]->shape(0));
  EXPECT_EQ(10, m_output_vec[0]->shape(1));
  float target[] = {
    0.144026, 0.107821, 0.079775, 0.058512, 0.042643, 0.070217, 0.074618, 0.079387, 0.084700, 0.090887,
    0.14903,  0.109821, 0.080564, 0.058821, 0.042763, 0.000236, 0.000460, 0.000899, 0.001758, 0.003427,
    0.149234, 0.109866, 0.080574, 0.058823, 0.042763, 7.13897E-7, 2.53995E-6, 9.0551E-6, 0.000032, 0.000116,
    0.149242, 0.109867, 0.080574, 0.058823, 0.042763, 2.16135E-9, 1.40115E-8, 9.10167E-8, 5.93344E-7, 3.88017E-6,
    0.149242, 0.109867, 0.080574, 0.058823, 0.042763, 6.52389E-12, 7.68712E-11, 9.05782E-10, 1.06728E-8, 1.25555E-7
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}
