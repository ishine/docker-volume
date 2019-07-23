#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
//  char buff[8] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', '\0'};
//  std::cout << '[' << buff << ']' << std::endl;
//  std::string name("this");
//  name.resize(8);
//  std::cout << name.length() << std::endl;
//  std::cout << "[" << name << "]" << std::endl;
//  std::cout << "[" << name.c_str() << "]" << std::endl;
//  std::cout << "[" << name.data() << "]" << std::endl;
//  memcpy(buff, name.data(), 32);
//  std::cout << '[' << buff << ']' << std::endl;
  FILE *fp = fopen("testdata-4-weight-simple.dat", "wb");
  uint16_t magic_num = 0x1;
  size_t res = fwrite(&magic_num, 1, 2, fp);
  if (2 != res) {
    std::cerr << "write magic number failed!" << std::endl;
    fclose(fp);
    return -1;
  }
  uint16_t tensor_num = 5;
  res = fwrite(&tensor_num, 1, 2, fp);
  if (2 != res) {
    std::cerr << "write tensor number failed!" << std::endl;
    fclose(fp);
    return -1;
  }
  const char* tensor_name[] = {
    "fc1.weight",
    "fc2.weight",
    "fc2.bias",
    "fc3.weight",
    "fc3.bias"
  };
  for (uint16_t i = 0; i < tensor_num; ++i) {
    std::string name(tensor_name[i]);
    name.resize(32);
    res = fwrite(name.data(), 1, 32, fp);
    if (32 != res) {
      std::cerr << "write tensor " << i << "'s name failed!" << std::endl;
      fclose(fp);
      return -1;
    }
  }
  uint32_t tensor_size[] = {
    10*5,
    25*10,
    25,
    1*25,
    1
  };
  for (uint16_t i = 0; i < tensor_num; ++i) {
    res = fwrite(&tensor_size[i], 1, 4, fp);
    if (4 != res) {
      std::cerr << "write tensor " << i << "'s size failed!" << std::endl;
      fclose(fp);
      return -1;
    }
  }
  uint32_t tensor_val_num = 351;
  float tensor_val[tensor_val_num];
  for (uint32_t i = 0; i < tensor_val_num; ++i) {
    tensor_val[i] = i * 1.0 / tensor_val_num;
  }
  res = fwrite(tensor_val, 4, tensor_val_num, fp);
  if (res != tensor_val_num) {
    std::cerr << "write tensor val failed!" << std::endl;
    fclose(fp);
    return -1;
  }
  fclose(fp);

  return 0;
}
