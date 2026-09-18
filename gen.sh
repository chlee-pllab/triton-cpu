#!/bin/bash
source .venv/bin/activate
export CC=$HOME/llvm-project_v/install/bin/clang
#python3 scripts/compile_qwen_kernels_riscv.py      # only needed if you re-run build_qwen_engine.py
export TRITON_LOCAL_LIBOMP_PATH=$HOME/.triton-native-libomp
export TRITON_ALWAYS_COMPILE=1
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=/home/chlee/triton-cpu/dummp_sta
python3 scripts/compile_qwen_kernels_riscv.py
#python3 scripts/gen_qwen_driver.py --host chlee@140.114.78.64
