#include "net.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <iostream>

#include "operator-factory.h"
#include "json/json.h"


namespace lnn {

Net::~Net() {
  if (NULL != m_weight_data) {
    delete [] m_weight_data;
    m_weight_data = NULL;
  }
  m_weight_tensors.clear();
  m_weight_tensor_name.clear();
  m_weight_tensor_name2id.clear();
  m_dynamic_tensor_name.clear();
  m_dynamic_tensor_name2id.clear();
  m_operator_name.clear();
  m_input_ids.clear();
  m_output_ids.clear();
  if (NULL != m_json) {
    delete m_json;
    m_json = NULL;
  }
}

bool Net::load(const char *model_file, const char *weight_file, const char *dir) {
#ifdef USE_GLOG
  int argc = 2;
  char** argv = new char*[argc];
  std::string flag_file;
  if (NULL == dir) flag_file = "./gflags-4-lnn";
  else flag_file = (std::string(dir) + "/gflags-4-lnn");
  std::ifstream in(flag_file.c_str());
  if (in.is_open()) {
    std::string flag_option = ("--flagfile=" + flag_file);
    argv[0] = const_cast<char*>("light-nn");
    argv[1] = const_cast<char*>(flag_option.c_str());
    google::ParseCommandLineFlags(&argc, &argv, false);
    in.close();
  }
  argv[0] = argv[1] = NULL;
  delete [] argv;
  google::InitGoogleLogging("light-nn");
#endif  // USE_GLOG
  std::string s_model_file, s_weight_file;
  if (NULL == dir) {
    s_model_file = model_file;
    s_weight_file = weight_file;
  } else {
    s_model_file = (std::string(dir) + "/" + model_file);
    s_weight_file = (std::string(dir) + "/" + weight_file);
  }
  // load model_file
  Json::Reader reader;
  std::ifstream ifs(s_model_file.c_str());
  if (!ifs.is_open()) {
    LOG(ERROR) << "Open model file [" << s_model_file << "] failed!" << std::endl;
    return false;
  }
  m_json = new Json::Value;
  if (!reader.parse(ifs, *m_json)) {
    LOG(ERROR) << "Parse model file [" << s_model_file << "] failed!" << std::endl;
    ifs.close();
    return false;
  }
  ifs.close();
  LOG(INFO) << m_json->toStyledString() << std::endl;
  // parse dependency of operators
  if (!parse_operators_dependency()) {
    LOG(ERROR) << "Parse dependency of operators failed!" << std::endl;
    return false;
  }
  // load weight_file
  // format: magic_number(2B), weight_tensor_num(2B) {, tensor_name(32B)}+
  //         {, tensor_size(4B)}+ {, tensor_data(4B)}+
  //
  //         tensor_name: operator_name_xxx
  FILE *fh = fopen(s_weight_file.c_str(), "rb");
  if (NULL == fh) {
    LOG(ERROR) << "Open weight file [" << s_weight_file << "] failed!" << std::endl;
    return false;
  }
  fseek(fh, 0, SEEK_END);
  uint32_t file_size = ftell(fh);
  rewind(fh); //文件复位到初始位置
  LOG(INFO) << "file_size: " << file_size << std::endl;
  char *content = new char[file_size];
  size_t bytes_read = fread(content, 1, file_size, fh);
  if (bytes_read != file_size) {
    LOG(ERROR) << "Read weight file [" << s_weight_file << "] failed!" << std::endl;
    fclose(fh);
    return false;
  }
  fclose(fh);
  char *cursor = content;
  // for now, we do NOT care about magic number
  //uint16_t magic_number = *(uint16_t *)cursor;
  cursor += 2;
  uint16_t num_weight_tensor = *(uint16_t *)cursor;
  LOG(INFO) << "weight_tensor_num: " << num_weight_tensor << std::endl;
  cursor += 2;
  m_weight_tensor_name.resize(num_weight_tensor);
  m_weight_tensor_name2id.clear();
//  LOG(INFO) << "weight_tensor_name" << std::endl;
  for (uint16_t i = 0; i < num_weight_tensor; ++i) {
	  //tensor name = 32 字节
    std::string tensor_name = std::string(cursor + i * 64, 64);
    // use 'c_str()' to strip multiple terminator if possible
	//char* strtemp = GetChar(tensor_name.c_str());
    //m_weight_tensor_name[i] = strtemp;
	m_weight_tensor_name[i] = tensor_name.c_str();
    m_weight_tensor_name2id.insert(std::make_pair(tensor_name.c_str(), i));
//    LOG(INFO) << m_weight_tensor_name[i] << std::endl;
  }
  cursor += num_weight_tensor * 64;
  std::vector<size_t> weight_tensor_size(num_weight_tensor);
  std::vector<size_t> weight_tensor_offset(num_weight_tensor);//weight_tensor_offset py3.6=0,py2.7=
//  LOG(INFO) << "weight_tensor_size" << std::endl;
  for (uint16_t i = 0; i < num_weight_tensor; ++i) {
    uint32_t tensor_size = *(uint32_t *)cursor;
    weight_tensor_size[i] = tensor_size;
    if (0 == i) weight_tensor_offset[i] = 0;
    else weight_tensor_offset[i] = weight_tensor_offset[i-1] + weight_tensor_size[i-1];
    cursor += 4;// weight elem 4 字节
//    LOG(INFO) << weight_tensor_size[i] << std::endl;
  }
  uint32_t total_tensor_size = weight_tensor_offset[num_weight_tensor-1]
    + weight_tensor_size[num_weight_tensor-1];
  m_weight_data = new float[total_tensor_size];
  memcpy(m_weight_data, cursor, total_tensor_size * 4);
  delete [] content;
  m_weight_tensors.resize(num_weight_tensor);
  for (size_t i = 0; i < num_weight_tensor; ++i) {
    m_weight_tensors[i].set_name(m_weight_tensor_name[i]);
    m_weight_tensors[i].set_data(m_weight_data + weight_tensor_offset[i], weight_tensor_size[i]); 
    LOG(INFO) << "weight_tensor_name: " << m_weight_tensor_name[i] << std::endl;
    LOG(INFO) << "weight_tensor_size: " << weight_tensor_size[i] << std::endl;
    LOG(INFO) << "weight_tensor_offset: " << weight_tensor_offset[i] << std::endl;
    // tensor shape has NOT been initialized, it will be set in operator which uses it
  }

  std::cout << "attribute value of net.cc:" << std::endl;
  std::cout << "m_weight_data" << std::endl;
  std::cout << total_tensor_size << std::endl;
  //for (int i = 0; i <total_tensor_size; i++)
  //{
	 // std::cout << m_weight_data[i]<<"\t ";
  //}
  std::cout << std::endl;
  std::cout << "num_weight_tensor:\t" << num_weight_tensor << std::endl;
  std::cout << "the size of m_dynamic_tensor_name:\n" << m_dynamic_tensor_name.size() << std::endl;
 /* for (int i = 0; m_dynamic_tensor_name.size(); i++)
  {
	  std::cout << *(m_dynamic_tensor_name.data() + i) << "\t";
  }*/
  //std::cout << "m_weight_data:\t" << *m_weight_data << std::endl;
  //std::cout << "m_weight_data:\t" << *m_weight_data << std::endl;
  //std::cout << "m_weight_data:\t" << *m_weight_data << std::endl;
  return true;
}

bool Net::parse_operators_dependency() {
  m_dynamic_tensor_name2id.clear();
  std::map<std::string, size_t>::iterator it;
  int op_num = (*m_json)["operators"].size();
  int idx = 0;
  m_operator_name.resize(op_num);
  for (int i = 0; i < op_num; ++i) {
    const Json::Value &op = (*m_json)["operators"][i];
    // process type & name
    if (!is_valid_operator(op["type"].asString())) {
      LOG(ERROR) << "Unknown operator type ["
        << op["type"].asString() << "]!" << std::endl;
      return false;
    }
    m_operator_name[i] = op["name"].asString();

    std::vector<size_t> input_ids, output_ids;


	std::cout <<"输入op的大小："<< op["input"].size() << std::endl;
    // process input
    for (int j = 0; j < static_cast<int>(op["input"].size()); ++j) {
      std::string name = op["input"][j].asString();
      it = m_dynamic_tensor_name2id.find(name);
      if (m_dynamic_tensor_name2id.end() == it) {
        m_dynamic_tensor_name2id.insert(std::make_pair(name, idx));
        idx++;
      }
      input_ids.push_back(m_dynamic_tensor_name2id[name]);
    }
    m_input_ids.push_back(input_ids);
    // process output
	std::cout << "输出op的大小：" << op["output"].size() << std::endl;
    for (int j = 0; j < static_cast<int>(op["output"].size()); ++j) {
      std::string name = op["output"][j].asString();
      it = m_dynamic_tensor_name2id.find(name);
      if (m_dynamic_tensor_name2id.end() == it) {
        m_dynamic_tensor_name2id.insert(std::make_pair(name, idx));
        idx++;
      }
      output_ids.push_back(m_dynamic_tensor_name2id[name]);
    }
    m_output_ids.push_back(output_ids);
  }

  m_dynamic_tensor_name.resize(idx);
  for (it = m_dynamic_tensor_name2id.begin();
       it != m_dynamic_tensor_name2id.end(); ++it) {
    m_dynamic_tensor_name[it->second] = it->first;
  }
  return true;
}

}  // namespace lnn
