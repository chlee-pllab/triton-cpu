"""Run one @triton.jit kernel on the native (x86) CPU backend, in a fresh
subprocess, and pickle back the post-call argument values.

This is invoked as a subprocess (not called in-process) specifically so its
triton/LLVM native-backend compilation state never shares a process with a
riscv64 cross-compile of the same kernel -- mixing "TRITON_CPU_TARGET=native"
and the riscv64 default target for the same @triton.jit function in one
process was observed to corrupt the heap (glibc "malloc(): ... corrupted",
consistently reproducible), most likely from stale/aliased backend/target
state cached across the two different compile targets.

Usage: native_kernel_runner.py <src_path> <kernel_name> <gx,gy,gz> <in_pkl> <out_pkl>
  <src_path>: a .py file containing "@triton.jit\ndef <kernel_name>(...): ..."
  <in_pkl>: pickled list of pre-call argument values, in arg_names order
  <out_pkl>: where to pickle the post-call argument values (same order)
"""
import os

os.environ["TRITON_DEFAULT_BACKEND"] = "cpu"
os.environ["TRITON_CPU_TARGET"] = "native"

import pickle
import sys

import triton
import triton.language as tl

triton.runtime.driver.set_active_to_cpu()
import torch  # noqa: F401  (needed for unpickling tensors, and by kernel source)


def main():
    src_path, name, grid_str, in_path, out_path = sys.argv[1:6]
    grid = tuple(int(v) for v in grid_str.split(","))

    ns = {"triton": triton, "tl": tl}
    try:
        from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
        ns["libdevice"] = libdevice
        ns["tl_math"] = tl_math
    except ImportError:
        pass
    with open(src_path) as f:
        exec(compile(f.read(), src_path, "exec"), ns)
    kernel = ns[name]

    with open(in_path, "rb") as f:
        call_args = pickle.load(f)

    kernel[grid](*call_args)

    with open(out_path, "wb") as f:
        pickle.dump(call_args, f)


if __name__ == "__main__":
    main()
