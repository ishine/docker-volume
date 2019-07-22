cmake_minimum_required(VERSION 2.8)

project(lnn)

set(LNN_VERSION_MAJOR 1)
set(LNN_VERSION_MINOR 0)

#让普通文件也能使用cmake中的变量
configure_file(
  ${PROJECT_SOURCE_DIR}/config.h.in
  ${PROJECT_BINARY_DIR}/config.h
  )
include_directories(${PROJECT_BINARY_DIR})
include_directories(include)
include_directories(third_party)
include_directories(third_party/openblas)

# --- [ Options
#   use cmake -DVAR=VALUE to set
#可选参数
option(DEBUG "Whether to debug the internal result" OFF)
option(USE_OPENMP "Whether use OpenMP to accelerate computation" ON)

if(USE_OPENMP)
  find_package(OpenMP)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
file(COPY ${PROJECT_SOURCE_DIR}/test/data
     DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ${PROJECT_SOURCE_DIR}/third_party/openblas/lib64/openblas.lib
     DESTINATION ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

file(GLOB_RECURSE GTEST_SRCS third_party/gtest/*.cc)
file(GLOB_RECURSE GTEST_HDRS third_party/gtest/*.h)
file(GLOB_RECURSE LNN_SRCS src/*.cc third_party/*.cc)
file(GLOB_RECURSE LNN_HDRS include/*.h third_party/*.h)
list(REMOVE_ITEM LNN_SRCS ${GTEST_SRCS})
list(REMOVE_ITEM LNN_HDRS ${GTEST_HDRS})

# --- [ Compile project library
add_library(lnn ${LNN_SRCS} ${LNN_HDRS})
target_link_libraries(lnn ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/openblas.lib)

# --- [ Compile gtest library for unit test
add_library(gtest ${GTEST_SRCS} ${GTEST_HDRS})
if(NOT USE_OPENMP)
  target_link_libraries(gtest pthread)
endif()

# --- [ Compile unit test
file(GLOB_RECURSE UT_SRCS test/test-*.cc)
foreach(f ${UT_SRCS})
  get_filename_component(name ${f} NAME_WE)
  add_executable(${name} ${f})
  target_link_libraries(${name} gtest lnn)
endforeach()
add_executable(test-all ${UT_SRCS})
target_link_libraries(test-all gtest lnn)
