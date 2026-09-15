import os
os.environ['TRITON_DEFAULT_BACKEND'] = 'cpu'
os.environ['TRITON_CPU_TARGET'] = 'native'
import triton
import triton.language as tl
triton.runtime.driver.set_active_to_cpu()
import torch
import torch._inductor.config as inductor_config
inductor_config.cpu_backend = 'triton'

import triton.backends.cpu.driver as cpu_driver

captured = {}
_orig_init = cpu_driver.CPULauncher.__init__
_orig_call = cpu_driver.CPULauncher.__call__


def capturing_init(self, src, metadata):
    _orig_init(self, src, metadata)
    self._capture_src = src
    self._capture_name = metadata.name if hasattr(metadata, "name") else metadata["name"]


def capture_arg(a):
    if not isinstance(a, torch.Tensor):
        return a
    # A kernel's index arithmetic computes offsets relative to a.data_ptr(),
    # and can validly reach anywhere within the tensor's real backing
    # allocation -- not just within a.numel() elements of it. Inductor often
    # hands a kernel a *view* (e.g. one half of a wider "cat" buffer, sliced
    # to fewer columns than its row stride): a.shape/.numel() then undercount
    # what's actually reachable from data_ptr(). A plain a.detach().clone()
    # repacks to a dense buffer sized by that undercount, so replaying the
    # kernel's real offsets against it is a genuine out-of-bounds write (this
    # reproducibly corrupted the heap: glibc "malloc(): ... corrupted").
    # Capture the full remaining flat storage from data_ptr() to the end of
    # the underlying allocation instead, so every offset the kernel could
    # legitimately touch stays in bounds.
    t = a.detach()
    elem_size = t.element_size()
    remaining_elems = t.untyped_storage().nbytes() // elem_size - t.storage_offset()
    flat = torch.empty(0, dtype=t.dtype)
    flat.set_(t.untyped_storage(), t.storage_offset(), (remaining_elems, ), (1, ))
    return flat.clone()


def capturing_call(self, *args, **kwargs):
    name = self._capture_name
    if name not in captured:
        kernel_args = args[9:]
        pre = [capture_arg(a) for a in kernel_args]
        captured[name] = {
            "src": self._capture_src,
            "grid": args[0:3],
            "pre_args": pre,
        }
    return _orig_call(self, *args, **kwargs)


cpu_driver.CPULauncher.__init__ = capturing_init
cpu_driver.CPULauncher.__call__ = capturing_call

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = os.path.join(os.environ["HOME"], "qwen2.5-0.5b-instruct")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cpu").eval()

compiled_model = torch.compile(model, backend="inductor")
prompt = "Please briefly explain what Time to First Token (TTFT) is."
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    compiled_model(**inputs)

cpu_driver.CPULauncher.__init__ = _orig_init
cpu_driver.CPULauncher.__call__ = _orig_call

print(f"Captured {len(captured)} distinct kernels")

MAX_ELEMS = 2_000_000  # skip kernels touching e.g. the full vocab embedding table --
# impractical to bake as C array literals, not a meaningful test of RVV codegen anyway.

to_deploy = []
for name, info in captured.items():
    max_elems = max((v.numel() for v in info["pre_args"] if isinstance(v, torch.Tensor)), default=0)
    if max_elems > MAX_ELEMS:
        print(f"SKIP (too large to bake as literals, max_single_tensor={max_elems}): {name}")
        continue
    to_deploy.append(name)

print("to_deploy order:", to_deploy)

debug_start = int(os.environ.get("DEBUG_DEPLOY_START", "0"))
debug_limit = int(os.environ.get("DEBUG_DEPLOY_LIMIT", "0"))
if debug_limit:
    to_deploy = to_deploy[debug_start:debug_start + debug_limit]

print(f"\nDeploying {len(to_deploy)} of {len(captured)} kernels to the board...\n")


REBUILT_KERNELS_DIR = os.path.join(os.environ["HOME"], "triton_kernel_capture", "rebuilt_kernels")


def write_kernel_source(name, info):
    # JITCallable.__init__ always sets .raw_src = inspect.getsourcelines(fn) --
    # including any decorator lines directly above the def (e.g. Inductor's own
    # outer @triton_heuristics.pointwise(...), when present) -- regardless of
    # whether .fn (the live underlying callable) later gets nulled out by
    # Inductor's async compile-worker pool reconstructing this JITFunction in
    # the main process. Skip straight to the inner "@triton.jit\ndef ...",
    # which is a self-contained kernel using only triton/tl (+ libdevice/
    # tl_math for transcendental ops, if present), and always (re)write it to
    # disk so both the native-reference subprocess and this process's riscv64
    # compile load the exact same source.
    src_lines = info["src"].fn.raw_src
    start = next(i for i, line in enumerate(src_lines) if line.strip().startswith("@triton.jit"))
    src_text = "".join(src_lines[start:])
    path = f"{REBUILT_KERNELS_DIR}/{name}.py"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(src_text)
    return path


def reconstruct_kernel(src_path, name):
    ns = {"triton": triton, "tl": tl}
    try:
        from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
        ns["libdevice"] = libdevice
        ns["tl_math"] = tl_math
    except ImportError:
        pass
    # @triton.jit's JITFunction.__init__ calls inspect.getsourcelines(fn), which
    # needs a real file on disk (linecache can't resolve a synthetic exec filename).
    with open(src_path) as f:
        exec(compile(f.read(), src_path, "exec"), ns)
    return ns[name]


# Compute ground-truth outputs by actually running each kernel on the trusted
# native (x86) backend with the exact captured pre-call inputs, rather than
# trusting the "post_args" snapshot grabbed mid-model-run. Inductor's memory
# planner reuses/aliases output buffers across many kernel calls (e.g. two
# sibling kernels each writing one half of a "cat" result into the same
# storage); a post_args snapshot taken around just one kernel's call can pick
# up bytes that a *different* kernel is responsible for, at offsets this
# kernel's own masked stores never touch -- a false-positive mismatch. Running
# the same kernel standalone here, from the same pre-call state, reproduces
# exactly what this kernel itself is supposed to do to that buffer.
#
# This runs in its own subprocess (scripts/native_kernel_runner.py), one per
# kernel, rather than in-process here: calling kernel[grid](...) in-process
# compiles this kernel through triton's normal JIT path under
# TRITON_CPU_TARGET=native, and doing that in the same process that will later
# triton.compile() the identical kernel for the riscv64 cross-target (below)
# reproducibly corrupted the heap (glibc "malloc(): ... corrupted", crashing
# partway into the riscv compile) -- almost certainly stale backend/target
# state cached across the two different compile targets for the same kernel.
# A fresh subprocess per kernel sidesteps that entirely.
import pickle
import subprocess
import sys

NATIVE_RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native_kernel_runner.py")

src_paths = {}
kernels = {}
native_expected = {}
for name in to_deploy:
    info = captured[name]
    src_path = write_kernel_source(name, info)
    src_paths[name] = src_path
    kernel = reconstruct_kernel(src_path, name)
    kernels[name] = kernel
    arg_names = kernel.arg_names
    grid = tuple(int(g) for g in info["grid"])

    in_path = f"{REBUILT_KERNELS_DIR}/{name}.in.pkl"
    out_path = f"{REBUILT_KERNELS_DIR}/{name}.out.pkl"
    with open(in_path, "wb") as f:
        pickle.dump(info["pre_args"], f)
    subprocess.run(
        [sys.executable, NATIVE_RUNNER, src_path, name, ",".join(str(g) for g in grid), in_path, out_path],
        check=True,
    )
    with open(out_path, "rb") as f:
        call_args = pickle.load(f)
    native_expected[name] = dict(zip(arg_names, call_args))

del os.environ["TRITON_CPU_TARGET"]  # default (riscv64) for the board deploy compile

from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-model-kernels"

results = {}
for name in to_deploy:
    info = captured[name]
    src = info["src"]
    kernel = kernels[name]
    arg_names = kernel.arg_names
    pre_args = info["pre_args"]
    native_post = native_expected[name]

    arguments = {}
    expected = {}
    constexprs = {}
    signature = {}
    for arg_name, pre_v in zip(arg_names, pre_args):
        ty = src.signature[arg_name]
        if ty == "constexpr":
            constexprs[arg_name] = pre_v
            continue
        signature[arg_name] = ty
        if isinstance(pre_v, torch.Tensor):
            # riscv.py declares bf16/fp16 buffers as native __bf16/_Float16 (clang
            # narrows a plain decimal float literal for us), so ordinary float values
            # from .tolist() are correct -- no bit-pattern conversion needed.
            arguments[arg_name] = pre_v.flatten().tolist()
            post_v = native_post[arg_name]
            if pre_v.shape == post_v.shape and torch.equal(pre_v, post_v):
                continue
            expected[arg_name] = post_v.flatten().tolist()
        else:
            arguments[arg_name] = pre_v

    grid = tuple(int(g) for g in info["grid"])
    print(f"=== {name} === grid={grid} nargs={len(arg_names)}")
    try:
        result = compile_deploy_and_run(
            kernel,
            arguments,
            grid,
            f"artifacts/riscv/model_kernels/{name}.elf",
            RISCV_HOST,
            constexprs=constexprs,
            signature=signature,
            expected=expected,
            atol=2e-2,
            remote_dir=RISCV_REMOTE_DIR,
        )
        ok = result.returncode == 0 and "PASS" in result.stdout
        results[name] = ok
        print(f"  -> {'PASS' if ok else 'FAIL'}  (returncode={result.returncode})")
        if not ok:
            print("  stdout:", result.stdout[-2000:])
            print("  stderr:", result.stderr[-2000:])
    except Exception as e:
        results[name] = False
        import traceback
        traceback.print_exc()
        print(f"  -> EXCEPTION: {e}")

print("\n===== SUMMARY =====")
for name in to_deploy:
    print(f"{'PASS' if results.get(name) else 'FAIL'}  {name}")
print(f"\n{sum(results.values())}/{len(to_deploy)} kernels passed on board "
      f"({len(captured) - len(to_deploy)} skipped as too large)")
