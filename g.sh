#!/bin/bash

export LLVM_BUILD_DIR=$HOME/llvm-project_v/install
source .venv/bin/activate
LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
  LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
  LLVM_SYSPATH=$LLVM_BUILD_DIR \
  pip install -vvv -e .

export CC=$HOME/llvm-project_v/install/bin/clang
source .venv/bin/activate
TRITON_ALWAYS_COMPILE=1 \
  TRITON_KERNEL_DUMP=1 \
  TRITON_DUMP_DIR=dump \
  TRITON_CPU_BACKEND=1 \
  TRITON_VSETVL_MINE=1 \
  TRITON_BRANCH_TAIL=1 \
  python3 python/tutorials/01-vector-add.py
