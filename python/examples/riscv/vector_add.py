import triton
import triton.language as tl
from triton.backends.cpu.riscv import standalone_kernel_cli


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def main() -> int:
    size = 256
    block_size = 64
    x = [float(i) for i in range(size)]
    y = [float(2 * i) for i in range(size)]
    return standalone_kernel_cli(
        add_kernel,
        arguments={
            "x_ptr": x,
            "y_ptr": y,
            "output_ptr": [0.0] * size,
            "n_elements": size,
        },
        constexprs={"BLOCK_SIZE": block_size},
        grid=(triton.cdiv(size, block_size), ),
        expected={"output_ptr": [a + b for a, b in zip(x, y)]},
        default_output="artifacts/riscv/vector_add.elf",
        default_host="chlee@140.114.78.64",
    )


if __name__ == "__main__":
    raise SystemExit(main())
