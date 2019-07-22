#include "operators/lookup-table.h"

#include "gtest/gtest.h"

class LookupTableTest : public testing::Test {
 protected:
  virtual void SetUp() {
    // construct json config
    m_config["name"] = "lookup-table";

    // construct weight & bias tensor
    int feature_size(100);
    m_weight_data = new float[feature_size];
    for (int i = 0; i < feature_size; ++i) {
      m_weight_data[i] = i;
    }
    std::vector<size_t> shape(1);
    shape[0] = feature_size;
    m_weights.resize(1);
    m_weights[0].set_name("lookup-table.weight");
    m_weights[0].set_data(m_weight_data, shape);
    m_name2id.clear();
    m_name2id.insert(std::make_pair("lookup-table.weight", 0));

    // construct input & output tensor vector
    m_input_tensor = new lnn::Tensor;
    m_output_tensor = new lnn::Tensor;
    m_input_vec.push_back(m_input_tensor);
    m_output_vec.push_back(m_output_tensor);
  }
  virtual void TearDown() {
    delete [] m_weight_data;
    delete m_input_tensor;
    delete m_output_tensor;
  }

  Json::Value m_config;
  float *m_weight_data;
  std::vector<lnn::Tensor> m_weights;
  std::map<std::string, size_t> m_name2id;
  lnn::Tensor *m_input_tensor;
  lnn::Tensor *m_output_tensor;
  std::vector<lnn::Tensor *> m_input_vec;
  std::vector<lnn::Tensor *> m_output_vec;
};

TEST_F(LookupTableTest, ValidInput) {
  lnn::LookupTable lookuptable(m_config);
  EXPECT_TRUE(lookuptable.set_weight(m_weights, m_name2id));

  float input_data[] = {
    1, 3, 0,
    10, 9, 7,
    2, 5, 6,
    12, 15, 11,

    21, 22, 25,
    31, 32, 36,
    41, 43, 44,
    50, 59, 52};
  std::vector<size_t> shape(3);
  shape[0] = 2;
  shape[1] = 4;
  shape[2] = 3;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(lookuptable.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(8, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(2, m_output_vec[0]->shape(0));
  EXPECT_EQ(4, m_output_vec[0]->shape(1));
  float target[] = {
    4., 26., 13., 38.,
    68., 99., 128., 161.
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(LookupTableTest, InvalidInput) {
  lnn::LookupTable lookuptable(m_config);
  EXPECT_TRUE(lookuptable.set_weight(m_weights, m_name2id));

  // index can not be negative or >= input_size
  float input_data[] = {1, -3, 12, 101};
  std::vector<size_t> shape(3);
  shape[0] = 1;
  shape[1] = 2;
  shape[2] = 2;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_FALSE(lookuptable.forward(m_input_vec, m_output_vec));
}
