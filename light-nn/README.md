[**Chinese Version**](README_cn.md)

# light-nn: a neural network inference library based on [OpenBLAS](https://www.openblas.net/)

## Background

Deep Learning algorithms are usually trained using **GPU** offline, but the online/production environment has no **GPU**. As a result, we have to across machine room or across city if we want to access a **GPU**-based service. Under these circumstances, we have to maintain at least two code base: the first is the gpu service, which usually written in Python, and the second is the online/production application, which usually written in C/C++.

We aim to develop a tool with the following features

## Features

 - written in pure C/C++
 - can be integrated with online applications in forms of libraries or source code
 - support multi-threads (only **one** copy of parameters in neural networks)
 - no need to install deep learning frameworks in online/production environment

## [Get Started](/wikis/Get_Started)

## [Docs](/wikis/Docs)

## [FAQ](/wikis/FAQ)

## How to contribute

If you want to add customized operators, you should

Step 1. add `header`、`source`, and `unit test` files in `include/operators`、`src/operators`, and `test` directories.

Step 2. modify `include/operator-factory.h` to add the customized operator.

Step 3. compile and run the unit test by `./build.sh; cd build/test; ./test-all`, all the unit tests should be passed.

Step 4. make a pull request with code review.

## Application

At present, it has been used in voice assistant, intelligent customer service, and recommendation projects.
Please let us know if your team wants to use the code.
