// Copyright (c) 2017, Tencent Inc.
// All Rights Reserved
//
// Author: Wenfeng Xuan <johnxuan@tencent.com>
//
#ifndef LNN_UTILS_LOG_H_
#define LNN_UTILS_LOG_H_

#ifdef USE_GLOG
#include "thirdparty/glog/logging.h"
#else
#include <string.h>
#include <iostream>

namespace lnn {

enum LogLevel {
  INFO = 0,
  WARNING,
  ERROR,
  FATAL
};

const char g_log_prefix[] = {'I', 'W', 'E', 'F'};

#define __FILENAME__ (strstr(__FILE__, "light-nn") ? strstr(__FILE__, "light-nn") : __FILE__)

#define LOG(LEVEL)                                               \
  std::cerr << g_log_prefix[LEVEL] << " " << __FILENAME__ << ":" \
            << __LINE__ << " "                                   \

}  // namespace lnn

#endif  // USE_GLOG

#endif  // LNN_UTILS_LOG_H_
