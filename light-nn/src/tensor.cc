#include "tensor.h"
#include "utils/log.h"
#include "utils/math-functions.h"

namespace lnn {

Tensor::~Tensor() {
  if (!b_external) delete [] m_data;
  m_data = NULL;
  m_shape.clear();
}

Tensor& Tensor::operator=(const Tensor& other) {
  if (this == &other) return *this;
  b_external = false;
  m_name = other.m_name;
  m_num_element = other.m_num_element;
  m_shape = other.m_shape;
  delete [] m_data;
  m_data = new float[m_num_element];
  lnn_copy(m_num_element, other.m_data, m_data);
  return *this;
}

void Tensor::set_data(float *data, size_t num_element) {
  m_data = data;
  m_num_element = num_element;
}

//shape 的维度>1时，m_num_element是总的元素个数
void Tensor::set_data(float *data, const std::vector<size_t> &shape) {
  m_data = data;
  m_shape = shape;
  m_num_element = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    m_num_element *= shape[i];
  }
}

size_t Tensor::canonical_axis(int axis) {
  if (axis < 0) {
//    LOG(WARNING) << "Negative axis [" << axis << "], set it to be the first "
//      << "non-negative number according to axis + k*num_axes()!" << std::endl;
    while (axis < 0) { axis += num_axes(); }
    return axis;
  } else if (0 <= axis && static_cast<size_t>(axis) < num_axes()) {
    return axis;
  } else {  // axis >= num_axes()
	  LOG(INFO)<<"tensor name" << m_name << std::endl;
	  std::cout << m_shape.size() << std::endl;
    LOG(WARNING) << "Axis [" << axis << "] exceeds number of axes ["
      << num_axes() << "], set it to be axis % num_axes()!" << std::endl;
    return axis % num_axes();
  }
}

size_t Tensor::shape(int axis) {
  return m_shape[canonical_axis(axis)];
}

size_t Tensor::count(size_t begin_axis, size_t end_axis) {
  size_t res = 1;
  for (size_t i = begin_axis; i < end_axis; ++i) { res *= m_shape[i]; }
  return res;
}

size_t Tensor::count(size_t axis) {
  return count(axis, num_axes());
}

bool Tensor::reshape(const std::vector<size_t> &shape) {
  size_t cnt = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    cnt *= shape[i];
  }
  if (cnt != size()) {
    LOG(ERROR) << "element number mismatch (" << cnt
      << ", " << size() << ")!\n";
    return false;
  }
  m_shape = shape;
  return true;
}

void Tensor::realloc(const std::vector<size_t> &shape) {
  size_t cnt = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    cnt *= shape[i];
  }
  if (cnt > size()) {
    if (NULL != m_data && !b_external) delete [] m_data;
    m_data = new float[cnt];
  }
  m_num_element = cnt;
  m_shape = shape;
  b_external = false;
}

void Tensor::dump(std::ostream &ofs, bool is_float) {
  for (size_t i = 0; i < size(); ++i) {
    if (0 != i) ofs << ' ';
    if (is_float) ofs << std::fixed << std::setprecision(6) << data()[i];
    else ofs << int(data()[i]);
  }
  ofs << std::endl;
}

void Tensor::get_data(float *data)
{
	data = m_data;
}

}  // namespace lnn
