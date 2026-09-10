#!/bin/bash

export LLVM_BUILD_DIR=$HOME/llvm-project_v/install
#source .venv/bin/activate
#LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
#  LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
#  LLVM_SYSPATH=$LLVM_BUILD_DIR \
#  pip install -vvv -e .

export CC=$HOME/llvm-project_v/install/bin/clang
source .venv/bin/activate

: "${TRITON_BENCH_WARMUP:=20}"
: "${TRITON_BENCH_ITERS:=200}"
export TRITON_BENCH_WARMUP TRITON_BENCH_ITERS

export TRITON_ALWAYS_COMPILE=1
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=dump
export TRITON_CPU_BACKEND=1
#export TRITON_VSETVL_MINE=1
#export TRITON_BRANCH_TAIL=1

python python/tutorials/rvv_01-vector-add_elf.py
#python scripts/run_on_riscv.py --host chlee@140.114.78.64 python/tutorials/rvv_01-vector-add_live.py
