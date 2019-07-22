cc_library(
  name = "lnn",
  srcs = [
    "src/executor.cc",
    "src/net.cc",
    "src/tensor.cc",
    "src/operators/attention.cc",
    "src/operators/concat.cc",
    "src/operators/conv1d.cc",
    "src/operators/crf.cc",
    "src/operators/eltwise.cc",
    "src/operators/embedding.cc",
    "src/operators/gelu.cc",
    "src/operators/gru.cc",
    "src/operators/linear.cc",
    "src/operators/lookup-table.cc",
    "src/operators/lstm.cc",
    "src/operators/pooling1d.cc",
    "src/operators/reduction.cc",
    "src/operators/relu.cc",
    "src/operators/reverse.cc",
    "src/operators/rnn-adapter.cc",
    "src/operators/scale.cc",
    "src/operators/sigmoid.cc",
    "src/operators/softmax.cc",
    "src/operators/tanh.cc",
    "src/operators/weight-wrapper.cc",
    "src/operators/weighted-sum.cc",
    "src/utils/math-functions.cc",
  ],
  defs = [
    "USE_GLOG",
    #"USE_OPENMP",
    #"DEBUG",
  ],
  deps = [
    "//thirdparty/glog:glog",
    "./third_party/json:jsoncpp",
    "./third_party/openblas:openblas",
  ],
  extra_cppflags = [
    "-Ilight-nn",
    "-Ilight-nn/include",
    "-Ilight-nn/third_party",
    "-Ilight-nn/third_party/openblas",
    "-fopenmp",
  ]
)

cc_binary(
  name = "test-example",
  srcs = ["examples/test.cc"],
  deps = [":lnn"],
  extra_cppflags = ["-Ilight-nn"],
  extra_linkflags = ["-fopenmp"],
)



cppflags_4_ut = [
  "-Ilight-nn",
  "-Ilight-nn/include",
  "-Ilight-nn/third_party",
]

data_4_ut = [
  ("test/data/gflags-4-lnn", "gflags-4-lnn"),
  ("test/data/testdata-4-net-simple.json", "data/testdata-4-net-simple.json"),
  ("test/data/testdata-4-net.json", "data/testdata-4-net.json"),
  ("test/data/testdata-4-weight-simple.dat", "data/testdata-4-weight-simple.dat"),
  ("test/data/testdata-4-weight.dat", "data/testdata-4-weight.dat"),
]

deps_4_ut = [
  ":lnn",
  "./third_party/gtest:gtest",
]

cc_test(
  name = "test-executor",
  srcs = [
    "test/test-executor.cc"
  ],
  deps = deps_4_ut,
  extra_cppflags = cppflags_4_ut,
  extra_linkflags = ["-fopenmp"],
  testdata = data_4_ut
)

cc_test(
  name = "test-net",
  srcs = [
    "test/test-net.cc"
  ],
  deps = deps_4_ut,
  extra_cppflags = cppflags_4_ut,
  extra_linkflags = ["-fopenmp"],
  testdata = data_4_ut
)

cc_test(
  name = "test-tensor",
  srcs = [
    "test/test-tensor.cc"
  ],
  deps = deps_4_ut,
  extra_cppflags = cppflags_4_ut,
  extra_linkflags = ["-fopenmp"],
)

cc_test(
  name = "test-op-attention",
  srcs = [
    "test/test-op-attention.cc"
  ],
  deps = deps_4_ut,
  extra_cppflags = cppflags_4_ut,
  extra_linkflags = ["-fopenmp"],
)

cc_test(
  name = "test-op-concat",
  srcs = [
    "test/test-op-concat.cc"
  ],
  deps = deps_4_ut,
  extra_cppflags = cppflags_4_ut,
  extra_linkflags = ["-fopenmp"],
)

cc_test(
  name = "test-op-conv1d",
  srcs = [
    "test/test-op-conv1d.cc"
  ],
  deps = deps_4_ut,
  extra_cppflags = cppflags_4_ut,
  extra_linkflags = ["-fopenmp"],
)

cc_test(
  name = "test-op-crf",
  srcs = [
    "test/test-op-crf.cc"
  ],
  deps = deps_4_ut,
  extra_cppflags = cppflags_4_ut,
  extra_linkflags = ["-fopenmp"],
)

cc_test(
  name = "test-op-eltwise",
  srcs = [
    "test/test-op-eltwise.cc"
  ],
  deps = deps_4_ut,
  extra_cppflags = cppflags_4_ut,
  extra_linkflags = ["-fopenmp"],
)

cc_test(
  name = "test-op-embedding",
  srcs = [
    "test/test-op-embedding.cc"
  ],
  deps = deps_4_ut,
  extra_cppflags = cppflags_4_ut,
  extra_linkflags = ["-fopenmp"],
)

cc_test(
  name = "test-op-gelu",
  srcs = [
    "test/test-op-gelu.cc"
  ],
  deps = deps_4_ut,
  extra_cppflags = cppflags_4_ut,
  extra_linkflags = ["-fopenmp"],
)

cc_test(
  name = "test-op-gru",
  srcs = [
    "test/test-op-gru.cc"
  ],
  deps = deps_4_ut,
  extra_cppflags = cppflags_4_ut,
  extra_linkflags = ["-fopenmp"],
)

cc_test(
  name = "test-op-linear",
  srcs = [
    "test/test-op-linear.cc"
  ],
  deps = deps_4_ut,
  extra_cppflags = cppflags_4_ut,
  extra_linkflags = ["-fopenmp"],
)

cc_test(
  name = "test-op-lookup-table",
  srcs = [
    "test/test-op-lookup-table.cc"
  ],
  deps = deps_4_ut,
  extra_cppflags = cppflags_4_ut,
  extra_linkflags = ["-fopenmp"],
)

cc_test(
  name = "test-op-lstm",
  srcs = [
    "test/test-op-lstm.cc"
  ],
  deps = deps_4_ut,
  extra_cppflags = cppflags_4_ut,
  extra_linkflags = ["-fopenmp"],
)

cc_test(
  name = "test-op-pooling1d",
  srcs = [
    "test/test-op-pooling1d.cc"
  ],
  deps = deps_4_ut,
  extra_cppflags = cppflags_4_ut,
  extra_linkflags = ["-fopenmp"],
)

cc_test(
  name = "test-op-reduction",
  srcs = [
    "test/test-op-reduction.cc"
  ],
  deps = deps_4_ut,
  extra_cppflags = cppflags_4_ut,
  extra_linkflags = ["-fopenmp"],
)

cc_test(
  name = "test-op-relu",
  srcs = [
    "test/test-op-relu.cc"
  ],
  deps = deps_4_ut,
  extra_cppflags = cppflags_4_ut,
  extra_linkflags = ["-fopenmp"],
)

cc_test(
  name = "test-op-reverse",
  srcs = [
    "test/test-op-reverse.cc"
  ],
  deps = deps_4_ut,
  extra_cppflags = cppflags_4_ut,
  extra_linkflags = ["-fopenmp"],
)

cc_test(
  name = "test-op-rnn-adapter",
  srcs = [
    "test/test-op-rnn-adapter.cc"
  ],
  deps = deps_4_ut,
  extra_cppflags = cppflags_4_ut,
  extra_linkflags = ["-fopenmp"],
)

cc_test(
  name = "test-op-scale",
  srcs = [
    "test/test-op-scale.cc"
  ],
  deps = deps_4_ut,
  extra_cppflags = cppflags_4_ut,
  extra_linkflags = ["-fopenmp"],
)

cc_test(
  name = "test-op-sigmoid",
  srcs = [
    "test/test-op-sigmoid.cc"
  ],
  deps = deps_4_ut,
  extra_cppflags = cppflags_4_ut,
  extra_linkflags = ["-fopenmp"],
)

cc_test(
  name = "test-op-softmax",
  srcs = [
    "test/test-op-softmax.cc"
  ],
  deps = deps_4_ut,
  extra_cppflags = cppflags_4_ut,
  extra_linkflags = ["-fopenmp"],
)

cc_test(
  name = "test-op-tanh",
  srcs = [
    "test/test-op-tanh.cc"
  ],
  deps = deps_4_ut,
  extra_cppflags = cppflags_4_ut,
  extra_linkflags = ["-fopenmp"],
)

cc_test(
  name = "test-op-weight-wrapper",
  srcs = [
    "test/test-op-weight-wrapper.cc"
  ],
  deps = deps_4_ut,
  extra_cppflags = cppflags_4_ut,
  extra_linkflags = ["-fopenmp"],
)

cc_test(
  name = "test-op-weighted-sum",
  srcs = [
    "test/test-op-weighted-sum.cc"
  ],
  deps = deps_4_ut,
  extra_cppflags = cppflags_4_ut,
  extra_linkflags = ["-fopenmp"],
)
