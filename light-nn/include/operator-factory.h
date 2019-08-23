// Copyright (c) 2017, Tencent Inc.
// All Rights Reserved
//
// Author: Wenfeng Xuan <johnxuan@tencent.com>
//
#ifndef LNN_OPERATOR_FACTORY_H_
#define LNN_OPERATOR_FACTORY_H_

#include <string>

#include "json/json.h"

#include "operators/attention.h"
#include "operators/concat.h"
#include "operators/conv1d.h"
#include "operators/conv2d.h"
#include "operators/batchnorm2d.h"
#include "operators/crf.h"
#include "operators/eltwise.h"
#include "operators/embedding.h"
#include "operators/gelu.h"
#include "operators/gru.h"
#include "operators/linear.h"
#include "operators/lookup-table.h"
#include "operators/lstm.h"
#include "operators/pooling1d.h"
#include "operators/pooling2d.h"
#include "operators/reduction.h"
#include "operators/residual.h"
#include "operators/relu.h"
#include "operators/leakyrelu.h"
#include "operators/reverse.h"
#include "operators/rnn-adapter.h"
#include "operators/scale.h"
#include "operators/sigmoid.h"
#include "operators/softmax.h"
#include "operators/tanh.h"
#include "operators/weighted-sum.h"
#include "operators/weight-wrapper.h"

namespace lnn {

static const char * g_op_types[] = {
  "Attention",
  "Concat",
  "Conv1D",
  "Conv2D",
  "BatchNorm2D",
  "CRF",
  "Eltwise",
  "Embedding",
  "GELU",
  "GRU",
  "Linear",
  "LookupTable",
  "LSTM",
  "Pooling1D",
  "Pooling2D",
  "Reduction",
  "Residual",
  "ReLU",
  "LeakyReLU",
  "Reverse",
  "RNNAdapter",
  "Scale",
  "Sigmoid",
  "Softmax",
  "Tanh",
  "WeightedSum",
  "WeightWrapper"
};

inline bool is_valid_operator(const std::string &type) {
  for (size_t i = 0; i < sizeof(g_op_types) / sizeof(g_op_types[0]); ++i) {
    if (g_op_types[i] == type) return true;
  }
  return false;
}

inline void get_operator(const std::string &type, const Json::Value &config, Operator *&op) {
  if ("Attention" == type) {
    op = new Attention(config);
  } else if ("Concat" == type) {
    op = new Concat(config);
  } else if ("Conv1D" == type) {
    op = new Conv1D(config);
  }else if ("Conv2D" == type) {
	op = new Conv2D(config);
  }else if ("BatchNorm2D" == type) {
	op = new BatchNorm2D(config);
  } else if ("CRF" == type) {
    op = new CRF(config);
  } else if ("Eltwise" == type) {
    op = new Eltwise(config);
  } else if ("Embedding" == type) {
    op = new Embedding(config);
  } else if ("GELU" == type) {
    op = new GELU(config);
  } else if ("GRU" == type) {
    op = new GRU(config);
  } else if ("Linear" == type) {
    op = new Linear(config);
  } else if ("LookupTable" == type) {
    op = new LookupTable(config);
  } else if ("LSTM" == type) {
    op = new LSTM(config);
  } else if ("Pooling1D" == type) {
    op = new Pooling1D(config);
  }else if ("Pooling2D" == type) {
	op = new Pooling2D(config);
  }else if ("Residual" == type) {
	op = new Residual(config);
  } else if ("Reduction" == type) {
    op = new Reduction(config);
  } else if ("ReLU" == type) {
    op = new ReLU(config);
  }else if ("LeakyReLU" == type) {
	op = new LeakyReLU(config);
  } else if ("Reverse" == type) {
    op = new Reverse(config);
  } else if ("RNNAdapter" == type) {
    op = new RNNAdapter(config);
  } else if ("Scale" == type) {
    op = new Scale(config);
  } else if ("Sigmoid" == type) {
    op = new Sigmoid(config);
  } else if ("Softmax" == type) {
    op = new Softmax(config);
  } else if ("Tanh" == type) {
    op = new Tanh(config);
  } else if ("WeightedSum" == type) {
    op = new WeightedSum(config);
  } else if ("WeightWrapper" == type) {
    op = new WeightWrapper(config);
  } else {
    op = NULL;
    std::cerr << "Unknown operator type [" << type << "]!" << std::endl;
  }
}

}  // namespace lnn

#endif  // LNN_OPERATOR_FACTORY_H_
