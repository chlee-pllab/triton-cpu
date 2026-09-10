#!/usr/bin/env python3
import argparse
import os
import runpy
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True, help="user@host of the riscv64 board, e.g. chlee@140.114.78.64")
    parser.add_argument("--remote-dir", default="~/triton-riscv-remote")
    parser.add_argument("script", help="path to the triton-cpu script to run unmodified")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    os.environ.setdefault("TRITON_CPU_BACKEND", "1")
    os.environ.setdefault("CC", os.path.expanduser("~/toolchain/bin/clang"))

    script_path = os.path.abspath(args.script)
    sys.path.insert(0, os.path.dirname(script_path))
    sys.argv = [script_path, *args.script_args]

    from triton.backends.cpu.riscv_remote_driver import install
    install(args.host, remote_dir=args.remote_dir)

    runpy.run_path(script_path, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
