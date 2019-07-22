#include "operators/embedding.h"

#include "gtest/gtest.h"

class EmbeddingTest : public testing::Test {
 protected:
  virtual void SetUp() {
    // construct json config
    int input_size(100), output_size(10);
    m_config["name"] = "embedding1";
    Json::Value param;
    param["input_size"] = input_size;
    param["output_size"] = output_size;
    m_config["param"] = param;

    // construct weight & bias tensor
    m_weight_data = new float[input_size*output_size];
    for (int i = 0; i < input_size*output_size; ++i) {
      m_weight_data[i] = i;
    }
    std::vector<size_t> shape(2);
    shape[0] = input_size;
    shape[1] = output_size;
    m_weights.resize(1);
    m_weights[0].set_name("embedding1.weight");
    m_weights[0].set_data(m_weight_data, shape);
    m_name2id.clear();
    m_name2id.insert(std::make_pair("embedding1.weight", 0));

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

TEST_F(EmbeddingTest, ValidInput) {
  lnn::Embedding embedding(m_config);
  EXPECT_TRUE(embedding.set_weight(m_weights, m_name2id));

  float input_data[] = {1, 3, 0, 10};
  std::vector<size_t> shape(1, 4);
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(embedding.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(40, m_output_vec[0]->size());
  EXPECT_EQ(2, m_output_vec[0]->num_axes());
  EXPECT_EQ(4, m_output_vec[0]->shape(0));
  EXPECT_EQ(10, m_output_vec[0]->shape(1));
  float target[] = {
    10.,  11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,
    30.,  31.,  32.,  33.,  34.,  35.,  36.,  37.,  38.,  39.,
    0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,
    100., 101., 102., 103., 104., 105., 106., 107., 108., 109.
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_NEAR(target[i], m_output_vec[0]->data()[i], 1E-6);
  }
}

TEST_F(EmbeddingTest, InvalidInput) {
  lnn::Embedding embedding(m_config);
  EXPECT_TRUE(embedding.set_weight(m_weights, m_name2id));

  // index can not be negative or >= input_size
  float input_data[] = {1, -3, 12, 101, 74};
  std::vector<size_t> shape(1, 5);
  m_input_tensor->set_data(input_data, shape);

  EXPECT_FALSE(embedding.forward(m_input_vec, m_output_vec));
}
