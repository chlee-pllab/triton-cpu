import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


class TimingStreamer(TextStreamer):
    """Records a wall-clock timestamp for each generated token (prompt excluded),
    so TTFT (time to the 1st token) and TPOT (average time per subsequent token)
    can be computed from real per-token timing instead of one aggregate duration
    for the whole generate() call."""

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, skip_prompt=True, **kwargs)
        self.token_times = []

    def put(self, value):
        # TextStreamer.put() is called once with the full prompt first (a no-op
        # when skip_prompt=True, and the call that flips next_tokens_are_prompt to
        # False), then once per newly generated token thereafter -- capture the
        # flag before super().put() mutates it so only real decode steps are timed.
        is_prompt_call = self.next_tokens_are_prompt
        super().put(value)
        if not is_prompt_call:
            self.token_times.append(time.perf_counter())

    def ttft_tpot(self, generation_start):
        if not self.token_times:
            return None, None
        ttft = self.token_times[0] - generation_start
        if len(self.token_times) > 1:
            tpot = (self.token_times[-1] - self.token_times[0]) / (len(self.token_times) - 1)
        else:
            tpot = None
        return ttft, tpot

# ==========================================
# Step 1: Force-switch Triton's default backend to CPU
# ==========================================
os.environ["TRITON_DEFAULT_BACKEND"] = "cpu"
os.environ["TORCH_COMPILE_DEBUG"] = "1"  # Enable debug output to view the generated C++/Triton code
# Hint: to see extremely detailed MLIR lowering, uncomment the line below
# os.environ["MLIR_ENABLE_DUMP"] = "1"

import triton
# Explicitly activate the Triton CPU driver
triton.runtime.driver.set_active_to_cpu()

# Route Inductor's CPU codegen through Triton (this fork's CPU backend targets RVV)
# instead of Inductor's default native C++/OpenMP codegen.
import torch._inductor.config as inductor_config
inductor_config.cpu_backend = "triton"

print("==== [1/4] Triton-CPU backend initialized successfully ====")

# ==========================================
# Step 2: Load the real Qwen2.5-0.5B model and tokenizer
# ==========================================
model_name = os.path.join(os.environ["HOME"], "qwen2.5-0.5b-instruct")

print(f"Loading model {model_name} (this may take a few minutes; a high-memory CPU server is recommended)...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# To avoid CPU memory OOM, and since triton-cpu currently has the best support for bfloat16, we use bfloat16
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)
model.eval()
print("==== [2/4] Model and weights loaded ====")

# ==========================================
# Step 3: Use torch.compile to hand off Attention graph optimization to Triton-CPU
# ==========================================
print("Compiling the model's control graph with torch.compile (target backend: Triton-CPU)...")
# Compile Qwen's core Attention or the whole model as a dynamic graph
# When torch.compile detects TRITON_DEFAULT_BACKEND=cpu, the underlying Inductor lowers ops to the Triton-CPU backend
#
# NOTE: `torch.compile(model)` returns an OptimizedModule, but `.generate()` is not
# defined on it -- attribute lookup falls through OptimizedModule.__getattr__ to the
# *original* uncompiled module's bound `.generate`, whose internal `self(...)` calls
# then run on the original eager module, never touching Dynamo/Inductor at all. That
# silently skipped compilation entirely (fast "success", empty torchinductor_cache,
# no torch_compile_debug/ dir). Compiling `model.forward` in place instead means
# `model.generate(...)` (calling the *same* model instance) actually goes through
# the compiled forward.
model.forward = torch.compile(model.forward, backend="inductor")
compiled_model = model
print("==== [3/4] Op compilation / graph optimization ready ====")

# ==========================================
# Step 4: Run end-to-end inference test (Prefill + Decode)
# ==========================================
prompt = "Please briefly explain what Time to First Token (TTFT) is."
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
prompt_len = input_ids.shape[1]

print(f"\n[Input Prompt]: {prompt}")
print(f"[Prompt token count]: {prompt_len} (this will trigger the 2D op in the Prefill stage)")

print("\n==== [4/4] Starting end-to-end inference ====")


def run_generate(label, streamer):
    start = time.perf_counter()
    with torch.no_grad():
        outputs = compiled_model.generate(
            **inputs,
            max_new_tokens=32,  # generate 32 tokens to test the Decode stage
            do_sample=False,    # greedy search ensures a stable benchmark
            use_cache=True,     # enable KV-cache rolling
            streamer=streamer,
        )
    elapsed = time.perf_counter() - start
    ttft, tpot = streamer.ttft_tpot(start)
    print(f"[{label}] total: {elapsed:.2f}s"
          + (f"  TTFT: {ttft:.2f}s" if ttft is not None else "  TTFT: n/a")
          + (f"  TPOT: {tpot * 1000:.1f}ms" if tpot is not None else "  TPOT: n/a"))
    return outputs, ttft, tpot


# Warmup: same prompt/shape as the measured run below, so the measured run
# reuses the exact same compiled kernels (no shape-triggered recompile) and
# pays none of torch.compile's one-time JIT cost. A model's TTFT/TPOT, as
# reported in serving papers, is a steady-state number -- compilation is a
# one-time setup cost paid once per process, not part of per-request latency,
# so it must be excluded rather than averaged into the reported figures.
print("Warmup run (JIT compile; discarded from reported TTFT/TPOT)...")
run_generate("warmup, discarded", TimingStreamer(tokenizer))

outputs, ttft, tpot = run_generate("measured", TimingStreamer(tokenizer))

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n" + "="*40)
print("[End-to-end generation result]:")
print(generated_text)
print("="*40)

print(f"\n[Steady-state performance]:")
print(f"TTFT (time to first token): {ttft:.2f} seconds" if ttft is not None else "TTFT: n/a (no tokens generated)")
print(f"TPOT (avg time per output token, tokens 2..N): {tpot * 1000:.1f} ms" if tpot is not None else "TPOT: n/a (fewer than 2 tokens generated)")
