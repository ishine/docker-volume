# light-nn: 基于 [OpenBLAS](https://www.openblas.net/) 的神经网络前向计算库

## 背景

众所周知，深度学习算法通常使用 **GPU** 进行训练，但是在线/生产环境中鲜有 **GPU** 可供使用。如果要访问基于 **GPU** 的服务，就要跨机房，或者跨城市，这会带来不必要的延时开销。同时，在这种情况下，我们不得不维护至少2套代码库：一个是通常用 Python 语言的 gpu 服务，一个是通常用 C/C++ 语言的在线/生产应用。

我们希望开发一个具有如下特点的工具

## 特点

 - 用纯 C/C++ 编写
 - 可以以库或者源代码的形式与在线应用集成
 - 支持多线程(内存中只有 **1** 份神经网络参数)
 - 在线/生产环境无需安装开源深度学习框架

## [上手](/wikis/Get_Started)

## [文档](/wikis/Docs)

## [FAQ](/wikis/FAQ)

## 贡献代码

如果你想添加自定义 operator，应该

第 1 步. 在 `include/operators`、`src/operators`、`test` 目录分别添加 `头文件`、`源代码`、`单元测试` 文件

第 2 步. 在 `include/operator-factory.h` 文件中添加自定义的 operator

第 3 步. 编译、运行单元测试：`./build.sh; cd build/test; ./test-all`，确保所有的单元测试都通过

第 4 步. 发起 pull request

## 应用落地

目前已应用于语音助手类(腾讯听听音箱、企鹅极光盒子、王者荣耀机器人)、智能客服类(腾讯游戏知几)、推荐类等项目。
如果你的团队开始使用项目代码请告知。