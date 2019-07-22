#include "operators/pooling1d.h"

#include "gtest/gtest.h"

class Pooling1DTest : public testing::Test {
 protected:
  virtual void SetUp() {
    // construct json config
    int input_size(3), kernel_size(3);
    m_config["name"] = "pooling1d";
    Json::Value param;
    param["input_size"] = input_size;
    param["kernel_size"] = kernel_size;
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

// for type max
TEST_F(Pooling1DTest, MaxDefault) {
  lnn::Pooling1D pooling1d(m_config);

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 5;
  shape[1] = 3;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(pooling1d.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(9, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(3, m_output_vec[0]->shape(0));
  EXPECT_EQ(3, m_output_vec[0]->shape(1));
  float target[] = {
    6.,  7.,  8.,
    9.,  10., 11.,
    12., 13., 14.
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(Pooling1DTest, MaxPadding) {
  m_config["param"]["padding"] = 1;
  lnn::Pooling1D pooling1d(m_config);

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 5;
  shape[1] = 3;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(pooling1d.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(15, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(5, m_output_vec[0]->shape(0));
  EXPECT_EQ(3, m_output_vec[0]->shape(1));
  float target[] = {
    3.,  4.,  5.,
    6.,  7.,  8.,
    9.,  10., 11.,
    12., 13., 14.,
    12., 13., 14.
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(Pooling1DTest, MaxStride) {
  m_config["param"]["stride"] = 2;
  lnn::Pooling1D pooling1d(m_config);

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 5;
  shape[1] = 3;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(pooling1d.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(6, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(2, m_output_vec[0]->shape(0));
  EXPECT_EQ(3, m_output_vec[0]->shape(1));
  float target[] = {
    6.,  7.,  8.,
    12., 13., 14.
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(Pooling1DTest, MaxDilation) {
  m_config["param"]["dilation"] = 2;
  lnn::Pooling1D pooling1d(m_config);

  float input_data[18];
  for (int i = 0; i < 18; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 6;
  shape[1] = 3;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(pooling1d.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(6, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(2, m_output_vec[0]->shape(0));
  EXPECT_EQ(3, m_output_vec[0]->shape(1));
  float target[] = {
    12., 13., 14.,
    15., 16., 17.
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(Pooling1DTest, MaxGlobal) {
  m_config["param"]["global_pooling"] = true;
  lnn::Pooling1D pooling1d(m_config);

  float input_data[18];
  for (int i = 0; i < 18; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 6;
  shape[1] = 3;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(pooling1d.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(3, m_output_vec[0]->size());
  EXPECT_EQ(1, m_output_vec[0]->num_axes());
  EXPECT_EQ(3, m_output_vec[0]->shape(0));
  float target[] = {
    15., 16., 17.
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

// for type mean
TEST_F(Pooling1DTest, MeanDefault) {
  m_config["param"]["type"] = "MEAN";
  lnn::Pooling1D pooling1d(m_config);

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 5;
  shape[1] = 3;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(pooling1d.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(9, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(3, m_output_vec[0]->shape(0));
  EXPECT_EQ(3, m_output_vec[0]->shape(1));
  float target[] = {
    3.,  4.,  5.,
    6.,  7.,  8.,
    9.,  10., 11.
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(Pooling1DTest, MeanPadding) {
  m_config["param"]["type"] = "MEAN";
  m_config["param"]["padding"] = 1;
  lnn::Pooling1D pooling1d(m_config);

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 5;
  shape[1] = 3;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(pooling1d.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(15, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(5, m_output_vec[0]->shape(0));
  EXPECT_EQ(3, m_output_vec[0]->shape(1));
  float target[] = {
    1.,  5./3, 7./3,
    3.,  4.,  5.,
    6.,  7.,  8.,
    9.,  10., 11.,
    7.,  23./3, 25./3
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(Pooling1DTest, MeanStride) {
  m_config["param"]["type"] = "MEAN";
  m_config["param"]["stride"] = 2;
  lnn::Pooling1D pooling1d(m_config);

  float input_data[15];
  for (int i = 0; i < 15; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 5;
  shape[1] = 3;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(pooling1d.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(6, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(2, m_output_vec[0]->shape(0));
  EXPECT_EQ(3, m_output_vec[0]->shape(1));
  float target[] = {
    3., 4., 5.,
    9, 10., 11.
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(Pooling1DTest, MeanDilation) {
  m_config["param"]["type"] = "MEAN";
  m_config["param"]["dilation"] = 2;
  lnn::Pooling1D pooling1d(m_config);

  float input_data[18];
  for (int i = 0; i < 18; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 6;
  shape[1] = 3;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(pooling1d.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(6, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(2, m_output_vec[0]->shape(0));
  EXPECT_EQ(3, m_output_vec[0]->shape(1));
  float target[] = {
    6., 7., 8.,
    9., 10., 11.
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(Pooling1DTest, MeanGlobal) {
  m_config["param"]["type"] = "MEAN";
  m_config["param"]["global_pooling"] = true;
  lnn::Pooling1D pooling1d(m_config);

  float input_data[18];
  for (int i = 0; i < 18; ++i) { input_data[i] = i; }
  std::vector<size_t> shape(2);
  shape[0] = 6;
  shape[1] = 3;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(pooling1d.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(3, m_output_vec[0]->size());
  EXPECT_EQ(1, m_output_vec[0]->num_axes());
  EXPECT_EQ(3, m_output_vec[0]->shape(0));
  float target[] = {
    45./6, 51./6, 57./6
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}
