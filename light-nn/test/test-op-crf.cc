#include "operators/crf.h"

#include "gtest/gtest.h"

class CRFTest : public testing::Test {
 protected:
  virtual void SetUp() {
    // construct json config
    int label_size(5);
    m_config["name"] = "crf";
    Json::Value param;
    param["label_size"] = label_size;
    m_config["param"] = param;

    // construct weight tensor for transition
    m_weight_data = new float[(label_size+1)*label_size];
    for (int i = 0; i < (label_size+1)*label_size; ++i) {
      if (i % 2 == 0) m_weight_data[i] = i;
      else m_weight_data[i] = -1 * i;
    }
    std::vector<size_t> shape(2);
    shape[0] = label_size + 1;
    shape[1] = label_size;
    m_weights.resize(1);
    m_weights[0].set_name("crf.weight");
    m_weights[0].set_data(m_weight_data, shape);
    m_name2id.clear();
    m_name2id.insert(std::make_pair("crf.weight", 0));

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

TEST_F(CRFTest, Basic) {
  lnn::CRF crf(m_config);
  EXPECT_TRUE(crf.set_weight(m_weights, m_name2id));

  int label_size(5), len(6);
  float *input_data = new float[len*label_size];
  for (int i = 0; i < len*label_size; ++i) {
    if (i % 2 == 0) input_data[i] = i * 1. / (len * label_size);
    else input_data[i] = -1. * i / (len * label_size);
  }
  std::vector<size_t> shape(2);
  shape[0] = len;
  shape[1] = label_size;
  m_input_tensor->set_data(input_data, shape);

  EXPECT_TRUE(crf.forward(m_input_vec, m_output_vec));
  EXPECT_EQ(6, m_output_vec[0]->size());
  EXPECT_EQ(1, m_output_vec[0]->num_axes());
  EXPECT_EQ(6, m_output_vec[0]->shape(0));
  int target[] = {
    3, 3, 3, 3, 3, 3
  };
  for (size_t i = 0; i < m_output_vec[0]->size(); ++i) {
    EXPECT_EQ(target[i], int(m_output_vec[0]->data()[i]));
  }
}
