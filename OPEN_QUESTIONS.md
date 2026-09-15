# Session handoff — open questions and unfinished threads

Written at the end of a long session so a fresh session can pick up without
re-deriving context. Covers: what's fixed and confirmed working, what's
mid-flight, and what the user asked that never got a real answer.

## How to resume the "run real Qwen kernels on the RISC-V board" work

Script: `scripts/capture_and_run_qwen_kernels.py` (copied here from a
session scratchpad so it survives). Run it like:

```bash
cd /home/chlee/triton-cpu
source .venv/bin/activate
export CC=$HOME/llvm-project_v/install/bin/clang
export OMP_NUM_THREADS=4
export TRITON_LOCAL_LIBOMP_PATH=$HOME/.triton-native-libomp
export DEBUG_DEPLOY_LIMIT=1   # only deploy the first N kernels; unset for all
rm -rf ~/.triton/cache /tmp/torchinductor_chlee artifacts/riscv/model_kernels
python3 scripts/capture_and_run_qwen_kernels.py
```

What it does: runs the real local `~/qwen2.5-0.5b-instruct` model once
through `torch.compile(backend="inductor")` in `TRITON_CPU_TARGET=native`
mode (so it actually executes, giving a correct reference), capturing every
distinct kernel's real arguments via a hook on `CPULauncher.__call__` in
`third_party/cpu/backend/driver.py` (not modified on disk — hooked at
runtime only). It then switches to the default riscv64 target and, for each
captured kernel small enough to bake as C array literals, cross-compiles +
deploys + runs it on the real board (140.114.78.64) via the existing
`riscv.py` pipeline, verifying against the real captured output.

## STILL OPEN: `out_ptr0[32]` verification mismatch

Latest real result (`qwen_debug7.log` in the old scratchpad, not copied —
rerun to reproduce): the first real kernel
(`triton_poi_fused_neg_slice_transpose_view_1`, a bf16 negation:
`out[j] = -in[i]`) now compiles, links, and runs on the board successfully,
but fails verification at `out_ptr0[32]`. This is **not** the earlier
buffer-aliasing false positive (that's fixed — see below). This is either:

- A real bf16 codegen bug in this project's own RVV lowering (this whole
  backend is under active development per recent commits like "finish
  vsetvl" — plausible), or
- A subtler bug still in `capture_and_run_qwen_kernels.py`'s data marshaling.

**Not yet done:** printing the actual vs. expected value at the failing
index to tell "tiny rounding difference" (maybe raise `atol`, currently
`2e-2`) apart from "genuinely wrong result" (real bug, needs deeper
investigation into the RVV bf16 store/truncate codegen). The generated
standalone C harness's comparison code (`riscv.py`, `generate_runner`) only
prints `verification failed: name[i]`, not the two values — easiest fix is
temporarily editing that fprintf to also print `arg_{index}[i]` and
`expected_{index}[i]`.

## RESOLVED: `g.sh` produced nothing in `torchinductor_cache` (`.ttir`/`.llir`/`.asm` missing)

**Root cause found and fixed**: `e2e.py` did
`compiled_model = torch.compile(model, backend="inductor")` and then called
`compiled_model.generate(...)`. But `torch.compile(model)` returns an
`OptimizedModule`, which does **not** define `.generate` itself —
`OptimizedModule.__getattr__` falls through to the *original* uncompiled
module's bound `.generate` method. Inside that method, `self(...)` refers to
the original eager module (the `self` the bound method was captured from),
not the compiled wrapper — so Dynamo/Inductor was **never invoked at all**.
This is a well-known HF+`torch.compile` foot-gun. Symptoms that gave it away:
whole `g.sh` run completing in ~1 second (real Triton-CPU compilation of a
Qwen-sized graph takes minutes), and zero output anywhere — no
`torch_compile_debug/` dir despite `TORCH_COMPILE_DEBUG=1`, and a completely
empty `torchinductor_cache` (not even `fxgraph`/`triton` subdirs).

**Fix** (applied in `e2e.py`): compile `model.forward` in place instead of
wrapping the whole module, then call `.generate()` on the *same* model
object:
```python
model.forward = torch.compile(model.forward, backend="inductor")
compiled_model = model
...
outputs = compiled_model.generate(...)   # now actually runs the compiled forward
```
Verified: rerunning `g.sh` now takes ~220s (real compile), and
`$HOME/torchinductor_cache/triton/None/<hash>/*.ttir`, `*.llir`, `*.asm` are
all populated with real Triton-CPU IR/codegen for each fused kernel
(`triton_poi_fused_...`, `triton_per_fused_...`, including the flash
attention and RMSNorm kernels).

**Also resolves the previously-separate "`TRITON_DUMP_DIR=dump` stays empty"
question**: that was never a distinct bug — `dump/` was empty for the same
reason `torchinductor_cache` was empty (no compilation ever ran). After this
fix, `dump/<hash>/*.ttir,*.ttcir,*.llir,*.tttcir,*.asm,*.so` are all
populated too. No separate root cause to chase there.

## RESOLVED + VERIFIED ON HOST: ExecuTorch export of the real Qwen2.5-0.5B model works

Confirmed for real (not just documentation-reading) in an isolated venv at
`/tmp/claude-1005/.../scratchpad/executorch-venv` (session-ephemeral —
recreate with `python3 -m venv ...` + `pip install executorch
optimum-executorch transformers` if resuming; kept fully separate from this
repo's own `.venv` to avoid any torch-version conflict with the
Triton-CPU/riscv work):

```bash
optimum-cli export executorch \
  --model ~/qwen2.5-0.5b-instruct --task text-generation --recipe xnnpack \
  --output_dir <out>                          # unquantized: model.pte, 2.5GB
optimum-cli export executorch \
  --model ~/qwen2.5-0.5b-instruct --task text-generation --recipe xnnpack \
  --use_custom_sdpa --use_custom_kv_cache --qlinear 8da4w --qembedding 8w \
  --output_dir <out>                          # quantized: model.pte, 415MB
```

Both exit 0, no errors. Both `.pte` files verified to genuinely load and
report correct model metadata via
`executorch.extension.pybindings.portable_lib._load_for_executorch`:
`vocab_size=151936`, `n_layers=24` — exactly Qwen2.5-0.5B-Instruct's real
config, not a stub. The quantized 415MB version comfortably fits the
board's 7.2GB free disk (the unquantized 2.5GB one technically fits too but
eats a third of it — prefer the quantized one for actual deployment).

## RESOLVED + VERIFIED ON REAL HARDWARE: executor_runner cross-compiled and runs on the BPI-F3

Full chain now proven end-to-end, genuinely (not just "should work"):

1. Cloned `pytorch/executorch` (tag v1.4.1, **must be named exactly
   `executorch`** — a known upstream quirk, checked in at `~/executorch`)
   with submodules (`git submodule update --init --recursive --depth 1`).
2. ExecuTorch already ships an official riscv64-linux-gnu toolchain file
   and CMake preset (`examples/riscv/`, "RISC-V Support RFC"
   pytorch/executorch#18991 — brand new, dated 2026). Its default assumes
   apt-installed `gcc-riscv64-linux-gnu` (needs sudo, not available here),
   so wrote a custom one instead,
   `examples/riscv/riscv64-clang-toolchain.cmake`, reusing this project's
   own proven riscv64 clang toolchain (same one `riscv.py` and the
   compiler-rt/openmp builds use: `~/llvm-project_v/install/bin/clang` +
   `--sysroot=~/toolchain/sysroot --gcc-toolchain=~/toolchain
   -march=rv64gcv -mabi=lp64d`).
3. `cmake --preset riscv64-linux -DCMAKE_TOOLCHAIN_FILE=<ours> -G Ninja`.
   **Important gotcha hit and fixed**: the preset's default-found
   `PYTHON_EXECUTABLE` (system `/usr/bin/python3.11`) doesn't have
   `torchgen` installed. A CMake `execute_process` that does
   `import torchgen; print(os.path.dirname(torchgen.__file__))` then
   silently gets an **empty string** on failure, and the next line,
   `file(GLOB_RECURSE _torchgen_srcs "${torchgen-out}/*.py")`, becomes
   `file(GLOB_RECURSE ... "/*.py")` — **a recursive glob of the entire
   filesystem** for every `.py` file on the machine, baked into
   `build.ninja` as bogus dependencies (47MB generated file, build failures
   like `needed by 'Functions.h', missing and no known rule to make it`
   for random unrelated paths like `~/tiramisu/...` or snap-internal
   files). Real fix: pass
   `-DPYTHON_EXECUTABLE=<venv-with-torchgen>/bin/python3` explicitly
   (`build.ninja` dropped to 562KB, configure went from 168s to 2s). Also
   switch off CMake's default Unix-Makefiles generator (its legacy
   per-file `depend` scanning stalled for 5+ minutes on one target,
   `portable_ops_lib`, with zero progress) in favor of `-G Ninja`, which
   has no such issue.
4. `cmake --build cmake-out-riscv-ninja --target executor_runner -j$(nproc)`
   → genuine riscv64 ELF (`ELF 64-bit LSB pie executable, UCB RISC-V, RVC,
   double-float ABI ... interpreter /lib/ld-linux-riscv64-lp64d.so.1`),
   3.2MB.
5. `qemu-riscv64-static` (extracted earlier this session) **SIGILL**s on
   this binary — likely an RVV ISA-version mismatch (that qemu build is
   6.2.0 from Ubuntu 22.04's `qemu-user-static`, probably predates solid
   RVV 1.0 emulation; our `-march=rv64gcv` target is RVV 1.0). Not pursued
   further — skipped qemu entirely and went straight to the real board,
   which is the actually-meaningful target anyway.
6. scp'd `executor_runner` + the tiny bundled "add" test (`.bpte`, via
   `examples/riscv/aot_riscv.py --model add`) to `140.114.78.64`, ran
   directly: **genuine PASS**, exact match against the bundled reference
   (`mean_absolute_error: 0.000000`, `max_absolute_error: 0.000000`,
   `Test_result: PASS`). Confirmed the *entire* chain — export → cross-
   compile → real hardware execution — works for real, not just in theory.

**Not yet done (natural next steps):**
1. Cross-compile a runner that actually drives Qwen2.5-0.5B (not just the
   toy "add" bundled test) — check `examples/models/llama/main.cpp`-style
   runner code for one that drives the tokenizer + generate loop; may need
   `EXECUTORCH_BUILD_EXTENSION_LLM`/similar preset options, and note the
   quantized Qwen export used `--recipe xnnpack` which needs GCC 14+ on
   riscv64 per `tools/cmake/preset/riscv64_linux.cmake` (our clang-based
   toolchain auto-skips XNNPACK; may need a non-XNNPACK/portable-ops-only
   Qwen export instead, or a newer build of GCC for riscv64).
2. scp the quantized `model.pte` (415MB) + tokenizer files to the board,
   run the LLM runner, verify real generated text end-to-end on-device.
3. Still not done: a host-side `forward()`/generate-loop sanity check of
   the `.pte` against `e2e.py`'s output, before bothering with the board
   for the actual LLM (cheap to do first, still skipped).

## RESOLVED (with a real, actionable answer): on-device PyTorch feasibility

The user pushed back hard on an earlier dismissal of on-device PyTorch,
pointing at how they build `iree.runtime` on the board — a **runtime-only**
CMake build (`-DIREE_BUILD_COMPILER=OFF -DIREE_HAL_DRIVER_LOCAL_TASK=ON`
etc.) rsynced and built directly on-device — as a reference point for
whether an analogous minimal/runtime-only PyTorch build is practical.

**Answer: yes, practical — via ExecuTorch, not full eager libtorch.**
ExecuTorch (`pytorch/executorch`) is PyTorch's own purpose-built on-device
inference runtime, and it is architecturally the *exact* analogue of
`iree.runtime`: `torch.export` your model ahead-of-time on the host (which
already has working PyTorch — this repo's `.venv`), producing a `.pte`
file; cross-compile ExecuTorch's C++ runtime once; ship just the runner
binary + `.pte` to the board. **No Python, no PyTorch install, on-device at
all.** This sidesteps the 7.2GB-free-disk constraint entirely — the
footprint is tiny (ExecuTorch's core runtime is documented at ~50KB) versus
however many GB a full libtorch build/install would need.

Evidence this is real, not speculative:
- A documented, working case study cross-compiled ExecuTorch for a RISC-V64
  board with RVV 1.0 (CanMV K230) — [ExecuTorch on RISC-V K230 Bring-Up](https://www.rt-rk.com/executorch-on-canmv230-v1-1-part-1/).
  That target was actually *harder* than our board (K230 uses a custom
  RT-Smart microkernel ABI, not Linux/glibc — needed
  `-DET_HAVE_PREAD=0` to work around a missing `pread()`). Our BPI-F3/K1
  board runs real Linux/glibc (Bianbu/Debian-based, confirmed earlier this
  session), which is the standard, well-trodden ExecuTorch cross-compile
  target — should be *more* straightforward than the K230 case, not less.
  On-device requirement in that case study: exactly two files, the compiled
  `executor_runner` binary and the model file. Loaded in ~22.8ms.
- Qwen2.5-0.5B-Instruct specifically confirmed exportable and loadable —
  see the verified-on-host section above, no longer just "should work."
- Full PyTorch *is* also separately buildable/gettable for riscv64 if ever
  needed for something ExecuTorch can't do: prebuilt wheels exist at
  [KumaTea/pytorch-riscv64](https://github.com/KumaTea/pytorch-riscv64)
  (PyTorch 2.4, Python 3.8–3.11 only — **board's Python is 3.12.3**, so
  these specific wheels won't install as-is; would need an older Python
  alongside, or a from-source build). Not needed for the ExecuTorch path
  above (no Python at all on-device there), just useful to know it's not a
  dead end if eager PyTorch is ever genuinely required on-device.

**Important scope note:** this is a *separate, parallel* track from the
Triton-CPU RVV validation work above. ExecuTorch has its own kernel/backend
system (XNNPACK, custom kernels, delegate backends) — by default it does
**not** use Triton-generated kernels at all. It answers "can Qwen run
natively on this board with real autoregressive generation happening
on-device" (yes). It does not by itself validate this repo's own Triton-CPU
RVV codegen (that's what `scripts/capture_and_run_qwen_kernels.py` is for).
The two could in principle be connected later (ExecuTorch supports
registering custom kernels/backends — a Triton-CPU-backed ExecuTorch
backend is a real but large undertaking, not attempted).

**Not yet done / next steps if pursued:**
1. Confirm Qwen2.5-0.5B (not just Qwen3.5) exports cleanly via ExecuTorch on
   this host (`pip install executorch` in a throwaway venv or this repo's
   `.venv`, try the export step only, no board involved yet).
2. Cross-compile ExecuTorch's runtime for riscv64/Linux/glibc using this
   project's existing `~/toolchain` (same toolchain already used for
   Triton-CPU's own riscv64 kernel builds) — by the K230 precedent this
   should be a standard CMake toolchain-file cross-compile, no exotic ABI
   issues expected.
3. Ship runner + `.pte` to the board, run, verify real generated text.

## Fixed and confirmed working this session (don't re-litigate these)

- `e2e.py` (local `~/qwen2.5-0.5b-instruct` model, `cpu_backend="triton"`)
  runs correctly end-to-end via `./g.sh` with `TRITON_CPU_TARGET=native` —
  real Triton-CPU kernels, real generated text, verified twice.
- `TRITON_CPU_TARGET=native` env var (`python/triton/runtime/build.py`,
  `python/src/llvm.cc`'s `createTargetMachine`) — riscv64 default path is
  byte-for-byte unaffected (verified via identical target-triple/flags in
  compile logs both before and after).
- `TRITON_LOCAL_LIBOMP_PATH` wired up for Linux in `build.py` (was
  previously Apple-only, silently ignored elsewhere) — minimal diff, no new
  auto-detection logic in Python. Points at `~/.triton-native-libomp/`
  (symlinks: `include/omp.h` → riscv sysroot's omp.h, pure API declarations,
  arch-agnostic; `lib/libomp.so` → `~/HOST_TOOLCHAIN`'s native x86 libomp.so
  — that toolchain's own omp.h is too old/broken with this project's clang).
  Set in `g.sh`.
- riscv64 vector-add tutorial re-verified PASS on real hardware after all
  build.py/llvm.cc changes (regression check).
- AOTInductor/`run_from_nativert`/`launcher.so` compile-time-stage thread:
  fully built, then fully reverted per explicit user request once it became
  clear Dynamo/Inductor's plain JIT path (what's actually needed) doesn't
  use any of that machinery. `driver.py` is back to byte-identical with
  origin; core `compiler.py`'s `add_stages` signature is back to original.
- `third_party/cpu/backend/riscv.py`:
  - `_c_literal` NaN/Inf handling (uninitialized output buffers can contain
    either; bare `repr(nan)+"f"` produced the invalid C token `nanf`).
  - bf16/fp16 buffers now declared as native `__bf16`/`_Float16` (clang
    narrows a plain decimal literal automatically) instead of a raw-bit
    `uint16_t` hack — per explicit user steer, referencing
    `~/triton-riscv-workspace/triton-riscv` commit `3c28abf` ("add f16").
- Built a real riscv64 **compiler-rt** (`libclang_rt.builtins-riscv64.a`,
  via `~/llvm-project_v/build-compiler-rt-riscv64`, same recipe as the
  existing `~/llvm-project_v/build-openmp-riscv64`) to supply
  `__truncsfbf2`/`__extendbfsf2` etc., which this riscv64 toolchain's
  `libgcc.a` lacks entirely. Copied next to `libgcc.a` so it's already on
  the default link search path; `build.py` links `-lclang_rt.builtins-riscv64`
  only in non-native (riscv64) mode. Replaced (and is strictly better than)
  an earlier hand-rolled weak-symbol version of these two functions, which
  is now removed from `riscv.py`.
- Kernel-rebuild fix for Inductor's async compile-worker pool: captured
  kernels' `JITFunction.fn` and `.__globals__` are `None` (worker pool
  reconstructs them without the live callable) for every kernel except the
  very first compiled. Fixed by re-`exec`ing the kernel's `raw_src` (stripped
  of Inductor's outer `@triton_heuristics.pointwise(...)`-style decorator,
  keeping just `@triton.jit\ndef ...`) into a fresh namespace — written to a
  real temp file first since `inspect.getsourcelines` (called inside
  `@triton.jit`) needs one.
- False-positive verification bug (my own script, not the compiler): was
  comparing *all* pointer args' post-execution state, including unmutated
  inputs — Inductor's memory planner can alias an input's storage with an
  unrelated output buffer, so an unmutated input's "post" snapshot could
  pick up someone else's write. Fixed: only verify buffers that actually
  changed between pre- and post-capture.
- `TORCHINDUCTOR_CACHE_DIR=$HOME/torchinductor_cache` set in `g.sh` (was
  defaulting under `/tmp`).

## Net files changed on disk (`git diff --stat` at time of writing)

`g.sh`, `python/src/llvm.cc`, `python/triton/runtime/build.py`,
`third_party/cpu/backend/riscv.py`, plus the pre-existing (not
session-caused) deletions of `python/tutorials/rvv_01-vector-add_live.py`,
`scripts/run_on_riscv.py`, `third_party/cpu/backend/riscv_remote_driver.py`.
`e2e.py` also edited (local model path, `cpu_backend="triton"`). New:
`scripts/capture_and_run_qwen_kernels.py`, this file.
