#!/bin/bash

export LLVM_BUILD_DIR=$HOME/llvm-project_v/install
#source .venv/bin/activate
#LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
#  LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
#  LLVM_SYSPATH=$LLVM_BUILD_DIR \
#  pip install -vvv -e .
ninja -C build/cmake.linux-x86_64-cpython-3.10 libtriton.so

export CC=$HOME/llvm-project_v/install/bin/clang
source .venv/bin/activate

: "${TRITON_BENCH_WARMUP:=20}"
: "${TRITON_BENCH_ITERS:=200}"
: "${OMP_NUM_THREADS:=4}"
export TRITON_BENCH_WARMUP TRITON_BENCH_ITERS OMP_NUM_THREADS

export TRITON_ALWAYS_COMPILE=1
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=dump
export TRITON_CPU_BACKEND=1
# Inductor's own compile cache (ttir/ttcir/llir/asm/so land here, not in dump/ --
# see TRITON_DUMP_DIR above) defaults under /tmp; keep it under $HOME instead so
# it survives reboots/tmp-clearing and is easy to find.
export TORCHINDUCTOR_CACHE_DIR=$HOME/torchinductor_cache
# Compile CPU-Triton kernels for this host (not the default riscv64 cross-compile
# target), so e2e.py can actually load and run them here instead of only compiling.
#export TRITON_CPU_TARGET=native
# This project's clang has no bundled OpenMP runtime for the host architecture.
# ~/.triton-native-libomp/{include,lib} symlinks in a working omp.h (from the riscv
# sysroot -- pure API declarations, arch-agnostic) and a native x86 libomp.so (from
# ~/HOST_TOOLCHAIN, whose own omp.h is too old to compile with this clang).
export TRITON_LOCAL_LIBOMP_PATH=$HOME/.triton-native-libomp
#export TRITON_VSETVL_MINE=1
#export TRITON_BRANCH_TAIL=1

#python python/tutorials/rvv_01-vector-add_elf.py
python scripts/e2e.py
