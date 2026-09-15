#!/bin/bash
source .venv/bin/activate
export CC=$HOME/llvm-project_v/install/bin/clang
#export TRITON_BENCH_WARMUP=20
#export TRITON_BENCH_ITERS=200
export OMP_NUM_THREADS=8
#export TRITON_ALWAYS_COMPILE=1
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=dump
export TRITON_CPU_BACKEND=1
export TORCHINDUCTOR_CACHE_DIR=$HOME/torchinductor_cache
export TRITON_LOCAL_LIBOMP_PATH=$HOME/.triton-native-libomp
python scripts/capture_and_run_qwen_kernels.py
