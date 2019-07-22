#!/bin/bash

if [ -d "build1" ]; then
  rm -rf build1
fi
if [ -d "dist" ]; then
  rm -rf dist
fi

mkdir build1
cd build1
cmake .. -DUSE_OPENMP=OFF -DDEBUG=OFF
make
mkdir -p dist/include
mkdir -p dist/lib
mkdir -p dist/lib64
mkdir -p dist/tools
cp ../include/executor.h dist/include
cp ../include/net.h dist/include
cp ../include/tensor.h dist/include
cp ../include/light-nn.h dist/include
cp ../tools/pytorch2lnn.py dist/tools
cp -rf ../examples dist/
cd dist/lib64
ar x ../../lib/liblnn.a
ar x ../../lib/libopenblas.a
ar crs liblnn.a *.o
cp liblnn.a ../lib/
rm -rf *.o
cd ../../
mv dist ..

