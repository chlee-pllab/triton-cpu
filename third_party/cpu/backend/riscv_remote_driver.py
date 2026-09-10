from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path
from typing import Sequence

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase
from triton._C.libtriton import llvm

from .driver import CPUDeviceInterface, CPUUtils, ty_to_cpp
from .riscv import Toolchain

_INT_CTYPES = {"int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t"}


def _as_pointer_buffer(value):
    if hasattr(value, "data_ptr") and hasattr(value, "numel"):
        return value.data_ptr(), value.numel() * value.element_size()
    if hasattr(value, "ctypes") and hasattr(value, "nbytes"):
        return value.ctypes.data, value.nbytes
    raise TypeError(f"Remote riscv driver needs a torch.Tensor or numpy.ndarray pointer argument, got {type(value)}")


def _scalar_literal(value, c_type: str) -> str:
    return str(int(value)) if c_type in _INT_CTYPES else repr(float(value))


def _generate_generic_harness(kernel_name: str, arg_c_types: Sequence[tuple[str, bool]]) -> str:
    extern_types = [("void*" if is_ptr else ty) for ty, is_ptr in arg_c_types] + ["uint32_t"] * 6
    decls, blob_decls, call_args, writebacks = [], [], [], []
    for i, (ty, is_ptr) in enumerate(arg_c_types):
        if is_ptr:
            decls.append(f'  const char *path_{i} = argv[argi++];')
            blob_decls.append(f'  Blob blob_{i} = read_blob(path_{i});')
            call_args.append(f"blob_{i}.buf")
            writebacks.append(f'  write_blob(path_{i}, blob_{i}.buf, blob_{i}.size);')
        else:
            parse = "strtoll(argv[argi++], NULL, 10)" if ty in _INT_CTYPES else "strtod(argv[argi++], NULL)"
            decls.append(f"  {ty} arg_{i} = ({ty}){parse};")
            call_args.append(f"arg_{i}")
    call_prefix = ", ".join(call_args)
    if call_prefix:
        call_prefix += ", "

    return "\n".join([
        "#include <stdint.h>",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <time.h>",
        "",
        "typedef struct { void *buf; long size; } Blob;",
        "",
        "static Blob read_blob(const char *path) {",
        '  FILE *f = fopen(path, "rb");',
        '  if (!f) { perror(path); exit(1); }',
        "  fseek(f, 0, SEEK_END);",
        "  long size = ftell(f);",
        "  fseek(f, 0, SEEK_SET);",
        "  void *buf = malloc(size > 0 ? size : 1);",
        "  if (fread(buf, 1, size, f) != (size_t)size) { perror(path); exit(1); }",
        "  fclose(f);",
        "  Blob b = { buf, size };",
        "  return b;",
        "}",
        "",
        "static void write_blob(const char *path, void *buf, long size) {",
        '  FILE *f = fopen(path, "wb");',
        '  if (!f) { perror(path); exit(1); }',
        "  fwrite(buf, 1, size, f);",
        "  fclose(f);",
        "}",
        "",
        f"extern void {kernel_name}({', '.join(extern_types)});",
        "",
        "int main(int argc, char **argv) {",
        "  int argi = 1;",
        "  uint32_t gridX = (uint32_t)strtoul(argv[argi++], NULL, 10);",
        "  uint32_t gridY = (uint32_t)strtoul(argv[argi++], NULL, 10);",
        "  uint32_t gridZ = (uint32_t)strtoul(argv[argi++], NULL, 10);",
        *decls,
        *blob_decls,
        "",
        "  long bench_iters = 0, bench_warmup = 0;",
        "  const char *bench_env;",
        '  if ((bench_env = getenv("TRITON_BENCH_ITERS"))) bench_iters = atol(bench_env);',
        '  if ((bench_env = getenv("TRITON_BENCH_WARMUP"))) bench_warmup = atol(bench_env);',
        "  if (bench_iters < 0) bench_iters = 0;",
        "  if (bench_warmup < 0) bench_warmup = 0;",
        "",
        "  for (long it = 0; it < bench_warmup; ++it)",
        "    for (uint32_t x = 0; x < gridX; ++x)",
        "      for (uint32_t y = 0; y < gridY; ++y)",
        "        for (uint32_t z = 0; z < gridZ; ++z)",
        f"          {kernel_name}({call_prefix}x, y, z, gridX, gridY, gridZ);",
        "",
        "  struct timespec t0, t1;",
        "  clock_gettime(CLOCK_MONOTONIC, &t0);",
        "  for (long it = 0; it < (bench_iters > 0 ? bench_iters : 1); ++it)",
        "    for (uint32_t x = 0; x < gridX; ++x)",
        "      for (uint32_t y = 0; y < gridY; ++y)",
        "        for (uint32_t z = 0; z < gridZ; ++z)",
        f"          {kernel_name}({call_prefix}x, y, z, gridX, gridY, gridZ);",
        "  clock_gettime(CLOCK_MONOTONIC, &t1);",
        "  if (bench_iters > 0) {",
        "    double ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (double)(t1.tv_nsec - t0.tv_nsec);",
        '    printf("Time: %.4f ms (mean of %ld iters, %ld warmup)\\n",',
        "           ns / 1e6 / bench_iters, bench_iters, bench_warmup);",
        "  }",
        "",
        *writebacks,
        "  return 0;",
        "}",
        "",
    ])


class _RemoteContext:

    def __init__(self, host: str, remote_dir: str, ssh_opts: Sequence[str], toolchain: Toolchain | None,
                 local_dir: str):
        self.host = host
        self.remote_dir = remote_dir
        self.ssh_opts = list(ssh_opts)
        self.toolchain = toolchain or Toolchain.from_env()
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self._so_bytes: dict[str, bytes] = {}
        self._exe_cache: dict[tuple, Path] = {}
        self._deployed: set[str] = set()
        self._remote_dir_ready = False

    def cache_kernel_so(self, name: str, so_bytes: bytes) -> None:
        self._so_bytes[name] = so_bytes

    def _ensure_remote_dir(self) -> None:
        if not self._remote_dir_ready:
            subprocess.run(["ssh", *self.ssh_opts, self.host, f"mkdir -p {self.remote_dir}"], check=True)
            self._remote_dir_ready = True

    def _scp(self, local_paths: Sequence[Path]) -> None:
        subprocess.run(["scp", *self.ssh_opts, *[str(p) for p in local_paths], f"{self.host}:{self.remote_dir}"],
                        check=True)

    def _scp_back(self, remote_names: Sequence[str]) -> None:
        remotes = [f"{self.host}:{self.remote_dir}/{n}" for n in remote_names]
        subprocess.run(["scp", *self.ssh_opts, *remotes, str(self.local_dir)], check=True)

    def get_executable(self, kernel_name: str, arg_c_types: Sequence[tuple[str, bool]]) -> str:
        key = (kernel_name, tuple(arg_c_types))
        exe_name = f"{kernel_name}_remote"
        if key not in self._exe_cache:
            so_bytes = self._so_bytes.get(kernel_name)
            if so_bytes is None:
                raise RuntimeError(f"No cached .so for kernel '{kernel_name}'; load_binary was never called for it")
            runner_source = _generate_generic_harness(kernel_name, arg_c_types)
            so_path = self.local_dir / f"lib{kernel_name}.so"
            runner_path = self.local_dir / f"{exe_name}.c"
            exe_path = self.local_dir / exe_name
            so_path.write_bytes(so_bytes)
            runner_path.write_text(runner_source)
            cmd = self.toolchain.compile_command([runner_path.name, so_path.name], exe_path.name,
                                                   extra_args=["-Wl,-rpath,$ORIGIN"])
            subprocess.run(cmd, check=True, cwd=self.local_dir)
            self._exe_cache[key] = exe_path

        exe_path = self._exe_cache[key]
        so_path = self.local_dir / f"lib{kernel_name}.so"
        if exe_name not in self._deployed:
            self._ensure_remote_dir()
            self._scp([exe_path, so_path])
            subprocess.run(["ssh", *self.ssh_opts, self.host, f"chmod +x {self.remote_dir}/{exe_name}"], check=True)
            self._deployed.add(exe_name)
        return exe_name

    def run(self, kernel_name: str, grid: tuple[int, int, int], call_args: list, arg_c_types: Sequence[tuple[str,
                                                                                                              bool]]):
        exe_name = self.get_executable(kernel_name, arg_c_types)
        self._ensure_remote_dir()

        argv = [str(grid[0]), str(grid[1]), str(grid[2])]
        ptr_info = []
        local_inputs = []
        for i, (value, (c_type, is_ptr)) in enumerate(zip(call_args, arg_c_types)):
            if is_ptr:
                address, nbytes = _as_pointer_buffer(value)
                remote_name = f"{kernel_name}_arg{i}.bin"
                local_path = self.local_dir / remote_name
                local_path.write_bytes(ctypes.string_at(address, nbytes))
                ptr_info.append((address, nbytes, local_path, remote_name))
                local_inputs.append(local_path)
                argv.append(remote_name)
            else:
                argv.append(_scalar_literal(value, c_type))

        if local_inputs:
            self._scp(local_inputs)

        env_prefix = ""
        for var in ("TRITON_BENCH_ITERS", "TRITON_BENCH_WARMUP"):
            if var in os.environ:
                env_prefix += f"{var}={os.environ[var]} "
        remote_cmd = f"cd {self.remote_dir} && {env_prefix}./{exe_name} " + " ".join(argv)
        result = subprocess.run(["ssh", *self.ssh_opts, self.host, remote_cmd], capture_output=True, text=True,
                                 check=True)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="")

        if ptr_info:
            self._scp_back([name for _, _, _, name in ptr_info])
            for address, nbytes, local_path, remote_name in ptr_info:
                data = (self.local_dir / remote_name).read_bytes()
                ctypes.memmove(address, data, min(nbytes, len(data)))


class RiscvRemoteUtils(CPUUtils):
    _context: _RemoteContext | None = None

    def load_binary(self, name, kernel, shared_mem, device):
        RiscvRemoteUtils._context.cache_kernel_so(name, kernel)
        return (None, 0, 0, 0, 0)


class RiscvRemoteLauncher:

    def __init__(self, src, metadata):
        self.arg_names = list(src.fn.arg_names)
        self.signature = dict(src.signature)
        self.kernel_name = metadata.name

    def __call__(self, gridX, gridY, gridZ, stream, function, packed_metadata, launch_metadata, enter_hook,
                 exit_hook, *args):
        call_args = []
        arg_c_types = []
        for name, value in zip(self.arg_names, args):
            ty = self.signature[name]
            if ty == "constexpr":
                continue
            is_ptr = ty[0] == "*"
            c_type = ty_to_cpp(ty[1:] if is_ptr else ty) if is_ptr else ty_to_cpp(ty)
            arg_c_types.append((c_type, is_ptr))
            call_args.append(value)
        RiscvRemoteUtils._context.run(self.kernel_name, (gridX, gridY, gridZ), call_args, arg_c_types)


class RiscvRemoteDriver(DriverBase):

    def __init__(self, host: str, *, remote_dir: str = "~/triton-riscv-remote", ssh_opts: Sequence[str] = (),
                 toolchain: Toolchain | None = None, local_dir: str = "artifacts/riscv/remote"):
        RiscvRemoteUtils._context = _RemoteContext(host, remote_dir, ssh_opts, toolchain, local_dir)
        self.utils = RiscvRemoteUtils()
        self.launcher_cls = RiscvRemoteLauncher
        super().__init__()

    def get_current_device(self):
        return 0

    def get_active_torch_device(self):
        import torch
        return torch.device("cpu", self.get_current_device())

    def get_current_stream(self, device):
        return 0

    def get_current_target(self):
        cpu_arch = llvm.get_cpu_tripple().split("-")[0]
        return GPUTarget("cpu", cpu_arch, 0)

    def get_device_interface(self):
        return CPUDeviceInterface()

    @staticmethod
    def is_active():
        return True

    def get_benchmarker(self):
        from triton.testing import do_bench

        def do_bench_cpu(*args, **kwargs):
            if 'measure_time_with_hooks' not in kwargs:
                kwargs['measure_time_with_hooks'] = True
            return do_bench(*args, **kwargs)

        return do_bench_cpu

    def get_empty_cache_for_benchmark(self):
        import torch
        cache_size = 512 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cpu')

    def clear_cache(self, cache):
        cache.zero_()

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)


def install(host: str, **kwargs) -> RiscvRemoteDriver:
    import triton

    remote = RiscvRemoteDriver(host, **kwargs)
    triton.runtime.driver.set_active(remote)
    triton.runtime.driver.set_active_to_cpu = lambda: triton.runtime.driver.set_active(remote)
    return remote
