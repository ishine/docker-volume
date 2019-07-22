#include "net.h"

#include "gtest/gtest.h"

TEST(NetTest, Interface) {
  lnn::Net net;
  EXPECT_TRUE(net.load("./data/testdata-4-net.json", "./data/testdata-4-weight.dat"));/*
  EXPECT_TRUE(net.load("./net.json", "./weight.dat"));*/

  // weight tensor part
  EXPECT_EQ(7, net.weight_tensor_number());
  const char* weight_tensor_name[] = {
    "embed_lookuptable",
    "lstm_xh_weight",
    "lstm_xh_bias",
    "lstm_hh_weight",
    "lstm_hh_bias",
    "fc_weight",
    "fc_bias"
  };
  const std::vector<std::string>& w_name = net.weight_tensor_name();
  EXPECT_EQ(7, w_name.size());
  for (size_t i = 0; i < w_name.size(); ++i) {
    EXPECT_STREQ(weight_tensor_name[i], w_name[i].c_str());
  }
  size_t weight_tensor_size[] = {
    100 * 1000,
    4 * 256 * 100,
    4 * 256,
    4 * 256 * 256,
    4 * 256,
    1 * 256,
    1
  };
  const std::vector<lnn::Tensor>& tensor = net.weight_tensors();
  EXPECT_EQ(7, tensor.size());
  for (size_t i = 0; i < tensor.size(); ++i) {
    EXPECT_STREQ(weight_tensor_name[i], const_cast<lnn::Tensor&>(
        tensor[i]).name().c_str());
    EXPECT_EQ(weight_tensor_size[i], const_cast<lnn::Tensor&>(tensor[i]).size());
  }

  // dynamic tensor part
  EXPECT_EQ(6, net.dynamic_tensor_number());
  const char* dynamic_tensor_name[] = {
    "sequence",
    "embedding",
    "lstm",
    "feature",
    "score",
    "prob"
  };
  const std::vector<std::string>& d_name = net.dynamic_tensor_name();
  EXPECT_EQ(6, d_name.size());
  for (size_t i = 0; i < d_name.size(); ++i) {
    EXPECT_STREQ(dynamic_tensor_name[i], d_name[i].c_str());
  }

  // operators' name part
  const char* operator_name[] = {
    "embed",
    "lstm",
    "adapter",
    "fc",
    "prob"
  };
  const std::vector<std::string>& o_name = net.operator_name();
  EXPECT_EQ(5, o_name.size());
  for (size_t i = 0; i < o_name.size(); ++i) {
    EXPECT_STREQ(operator_name[i], o_name[i].c_str());
  }

  // operators' input & output part
  size_t input_target[] = {
    0,
    1,
    2,
    3,
    4
  };
  size_t output_target[] = {
    1,
    2,
    3,
    4,
    5
  };
  const std::vector<std::vector<size_t> >& input_ids = net.op_input_ids();
  const std::vector<std::vector<size_t> >& output_ids = net.op_output_ids();
  EXPECT_EQ(5, input_ids.size());
  EXPECT_EQ(5, output_ids.size());
  for (size_t i = 0; i < input_ids.size(); ++i) {
    EXPECT_EQ(1, input_ids[i].size());
    EXPECT_EQ(1, output_ids[i].size());
    EXPECT_EQ(input_target[i], input_ids[i][0]);
    EXPECT_EQ(output_target[i], output_ids[i][0]);
  }
}
