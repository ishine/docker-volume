#!/bin/bash

if [ 3 != $# ]; then
  echo "usage: $0 only_cblas use_openmp num_threads"
  exit
fi

PROJ="OpenBLAS"
VER="0.2.20"

CMP_DIR=$PROJ"-"$VER
if [ -d $CMP_DIR ]; then
  rm -rf $CMP_DIR
fi

tar -xf $CMP_DIR".tar.gz"
cd $CMP_DIR
make ONLY_CBLAS=$1 USE_OPENMP=$2 NUM_THREADS=$3
cd ..
cp $CMP_DIR/libopenblas.a ./lib64/
cp $CMP_DIR/libopenblas.so ./lib64/
rm -rf $CMP_DIR

