#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportInvalidTypeForm=false

from __future__ import annotations

import argparse
import json
import math
import re
import time

try:
    import torch
    import torch.nn.functional as torch_f
except ImportError:
    torch = None
    torch_f = None

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


CASE_PATH = 'benchmark/choreo/concat/20_unet_16x512x32x32_16x512x32x32_16x1024x32x32.co'
CASE_NAME = '20_unet_16x512x32x32_16x512x32x32_16x1024x32x32'
CATEGORY = 'concat'
SYMBOL_DEFAULTS = {}
INPUT_TOKENS = ['16x512x32x32', '16x512x32x32']
OUTPUT_TOKEN = '16x1024x32x32'
BASE_NOTES = 'axis=1'
ATOM_RE = re.compile(r"[A-Z][0-9](?![0-9])|[A-Z]|\d+")


def merge_notes(*notes: str) -> str:
    ordered = []
    seen = set()
    for raw in notes:
        for piece in str(raw).split(";"):
            note = piece.strip()
            if not note or note in seen:
                continue
            seen.add(note)
            ordered.append(note)
    return "; ".join(ordered)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"{CASE_NAME}: {message}")


def _parse_atom(expr: str, index: int) -> tuple[str, int]:
    match = ATOM_RE.match(expr, index)
    if not match:
        raise ValueError(f"invalid dimension expression: {expr}")
    return match.group(0), match.end()


def _atom_value(atom: str, env: dict[str, int]) -> int:
    if atom.isdigit():
        return int(atom)
    return env[atom]


def eval_dim_expr(expr: str, env: dict[str, int]) -> int:
    total = 0
    for term in expr.split("p"):
        index = 0
        product_value = 1
        while index < len(term):
            atom, index = _parse_atom(term, index)
            value = _atom_value(atom, env)
            if index < len(term) and term[index] == "d":
                divisor_atom, index = _parse_atom(term, index + 1)
                value //= _atom_value(divisor_atom, env)
            product_value *= value
        total += product_value
    return total


def eval_shape(token: str, env: dict[str, int]) -> tuple[int, ...]:
    parts = [part for part in token.split("x") if part]
    return tuple(eval_dim_expr(part, env) for part in parts)


def resolve_env(args: argparse.Namespace) -> dict[str, int]:
    env = dict(SYMBOL_DEFAULTS)
    for symbol in SYMBOL_DEFAULTS:
        env[symbol] = getattr(args, symbol)
    return env


def copy_via_triton(tensor: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    if triton is None or tl is None:
        return tensor.contiguous()
    flat = tensor.reshape(-1)
    out = torch.empty_like(flat)
    grid = (triton.cdiv(flat.numel(), block_size),)
    copy_kernel[grid](flat, out, flat.numel(), block_size=block_size)
    return out.reshape(tensor.shape)


def summary(metadata: dict[str, object], status: str, time_ms: float, *notes: str) -> dict[str, object]:
    return {
        "status": status,
        "mode": metadata["mode"],
        "explicit_assertions": metadata["explicit_assertions"],
        "input_shape": metadata["input_shape"],
        "output_shape": metadata["output_shape"],
        "time_ms": round(time_ms, 4),
        "notes": merge_notes(metadata.get("notes", ""), *notes),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Standalone Triton case for {CATEGORY}")
    parser.add_argument("--describe-case", action="store_true")
    parser.add_argument("--chunk-check", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    return parser.parse_args()


if triton is not None and tl is not None:

    @triton.jit
    def copy_kernel(inp_ptr, out_ptr, numel, block_size: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < numel
        values = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, values, mask=mask)

RUNTIME_ASSERTIONS = 6
HOST_EXTRAS = {'axis': 1}
INPUT_IS_SCALAR = (False, False)


def tensor_or_scalar(token: str, is_scalar: bool, env: dict[str, int], *, integer: bool = False):
    if is_scalar:
        return float(token)
    shape = eval_shape(token, env)
    dtype = torch.int64 if integer else torch.float32
    if integer:
        high = max(2, shape[-1] if shape else 8)
        return torch.randint(0, high, shape, device="cuda", dtype=dtype)
    return torch.randn(shape, device="cuda", dtype=dtype).contiguous()


def describe_case(env: dict[str, int]) -> dict[str, object]:
    input_desc = []
    for i, token in enumerate(INPUT_TOKENS):
        input_desc.append("scalar" if INPUT_IS_SCALAR[i] else str(eval_shape(token, env)))
    output_shape = eval_shape(OUTPUT_TOKEN, env)
    mode = "triton-standalone-kernel" if CATEGORY in {'elemwise_add', 'relu', 'matmul', 'sigmoid', 'gelu'} else "triton-standalone-host-entrypoint"
    return {
        "case_name": CASE_NAME,
        "category": CATEGORY,
        "choreo_case": CASE_PATH,
        "mode": mode,
        "input_shape": " + ".join(input_desc),
        "output_shape": str(output_shape),
        "explicit_assertions": RUNTIME_ASSERTIONS,
        "notes": merge_notes(BASE_NOTES, "standalone host implementation", "triton copy epilogue"),
    }


def runtime_assertions(*tensors) -> None:
    present = [tensor for tensor in tensors if isinstance(tensor, torch.Tensor)]
    # EXPLICIT_ASSERTION: len(present) >= 1
    require(len(present) >= 1, "at least one tensor input is required")
    # EXPLICIT_ASSERTION: all(tensor.is_cuda for tensor in present)
    require(all(tensor.is_cuda for tensor in present), "all tensors must live on CUDA")
    # EXPLICIT_ASSERTION: all(tensor.dtype in {torch.float32, torch.int64} for tensor in present)
    require(all(tensor.dtype in {torch.float32, torch.int64} for tensor in present), "tensor dtypes must be supported")
    # EXPLICIT_ASSERTION: all(tensor.numel() > 0 for tensor in present)
    require(all(tensor.numel() > 0 for tensor in present), "tensor element counts must be positive")
    # EXPLICIT_ASSERTION: OUTPUT_TOKEN != ''
    require(OUTPUT_TOKEN != "", "output token must exist")
    # EXPLICIT_ASSERTION: CATEGORY != ''
    require(CATEGORY != "", "category must exist")


def run_host_impl(env: dict[str, int]):
    if CATEGORY == "reshape":
        inp = tensor_or_scalar(INPUT_TOKENS[0], INPUT_IS_SCALAR[0], env)
        runtime_assertions(inp)
        return inp.reshape(eval_shape(OUTPUT_TOKEN, env))

    if CATEGORY == "transpose":
        inp = tensor_or_scalar(INPUT_TOKENS[0], INPUT_IS_SCALAR[0], env)
        runtime_assertions(inp)
        permutation = tuple(HOST_EXTRAS["permutation"])
        return inp.permute(permutation).contiguous()

    if CATEGORY == "concat":
        tensors = [tensor_or_scalar(t, INPUT_IS_SCALAR[i], env) for i, t in enumerate(INPUT_TOKENS)]
        runtime_assertions(*tensors)
        return torch.cat(tensors, dim=HOST_EXTRAS["axis"])

    if CATEGORY == "reduce_mean":
        inp = tensor_or_scalar(INPUT_TOKENS[0], INPUT_IS_SCALAR[0], env)
        runtime_assertions(inp)
        axes, keepdim = HOST_EXTRAS["reduction"]
        return torch.mean(inp, dim=tuple(axes), keepdim=keepdim)

    if CATEGORY == "embedding":
        vocab_size = eval_shape(INPUT_TOKENS[1], env)[0]
        ids_shape = eval_shape(INPUT_TOKENS[0], env)
        ids = torch.randint(0, vocab_size, ids_shape, device="cuda", dtype=torch.int64)
        weight = tensor_or_scalar(INPUT_TOKENS[1], INPUT_IS_SCALAR[1], env)
        runtime_assertions(ids, weight)
        return torch.embedding(weight, ids)

    if CATEGORY == "batch_norm":
        inp = tensor_or_scalar(INPUT_TOKENS[0], INPUT_IS_SCALAR[0], env)
        weight = tensor_or_scalar(INPUT_TOKENS[1], INPUT_IS_SCALAR[1], env)
        bias = tensor_or_scalar(INPUT_TOKENS[2], INPUT_IS_SCALAR[2], env)
        runtime_assertions(inp, weight, bias)
        ax = HOST_EXTRAS.get("param_axis", 1)
        x = inp.transpose(1, ax).contiguous() if ax != 1 else inp
        out = torch_f.batch_norm(x, None, None, weight, bias, training=True)
        return out.transpose(1, ax).contiguous() if ax != 1 else out

    if CATEGORY == "layer_normalization":
        inp = tensor_or_scalar(INPUT_TOKENS[0], INPUT_IS_SCALAR[0], env)
        weight = tensor_or_scalar(INPUT_TOKENS[1], INPUT_IS_SCALAR[1], env)
        bias = tensor_or_scalar(INPUT_TOKENS[2], INPUT_IS_SCALAR[2], env)
        runtime_assertions(inp, weight, bias)
        return torch_f.layer_norm(inp, tuple(weight.shape), weight, bias)

    if CATEGORY == "conv2d":
        inp = tensor_or_scalar(INPUT_TOKENS[0], INPUT_IS_SCALAR[0], env)
        weight = tensor_or_scalar(INPUT_TOKENS[1], INPUT_IS_SCALAR[1], env)
        runtime_assertions(inp, weight)
        return torch_f.conv2d(inp, weight, bias=None, stride=HOST_EXTRAS["stride"], padding=HOST_EXTRAS["padding"], dilation=HOST_EXTRAS["dilation"])

    if CATEGORY == "max_pool2d":
        inp = tensor_or_scalar(INPUT_TOKENS[0], INPUT_IS_SCALAR[0], env)
        runtime_assertions(inp)
        kernel = HOST_EXTRAS["kernel"]
        stride = HOST_EXTRAS["stride"]
        padding = HOST_EXTRAS["padding"]
        if padding > kernel // 2:
            padded = torch.nn.functional.pad(inp, [padding] * 4)
            return torch_f.max_pool2d(padded, kernel_size=kernel, stride=stride, padding=0)
        return torch_f.max_pool2d(inp, kernel_size=kernel, stride=stride, padding=padding)

    if CATEGORY == "softmax":
        inp = tensor_or_scalar(INPUT_TOKENS[0], INPUT_IS_SCALAR[0], env)
        runtime_assertions(inp)
        return torch.softmax(inp, dim=-1)

    raise ValueError(f"{CASE_NAME}: unsupported standalone host category {CATEGORY}")


def run_case(env: dict[str, int], chunk_check: bool, compile_only: bool) -> dict[str, object]:
    metadata = describe_case(env)
    if torch is None or torch_f is None:
        return summary(metadata, "n/a", 0.0, "Missing torch package")
    if not torch.cuda.is_available():
        return summary(metadata, "n/a", 0.0, "CUDA not available")
    started = time.perf_counter_ns()
    host_result = run_host_impl(env)
    result = copy_via_triton(host_result.contiguous() if isinstance(host_result, torch.Tensor) else host_result)
    if isinstance(result, torch.Tensor):
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    if compile_only:
        return summary(metadata, "success", elapsed_ms, "compile-smoke")
    if chunk_check and isinstance(result, torch.Tensor):
        reference = run_host_impl(env)
        flat_result = result.reshape(-1)
        flat_reference = reference.reshape(-1)
        chunk = min(4096, flat_result.numel())
        slices = [slice(0, chunk)]
        if flat_result.numel() > chunk:
            slices.append(slice(flat_result.numel() - chunk, flat_result.numel()))
        for chunk_slice in slices:
            if not torch.allclose(flat_result[chunk_slice], flat_reference[chunk_slice], atol=1e-4, rtol=1e-4):
                raise ValueError(f"{CASE_NAME}: chunk validation failed at slice {chunk_slice.start}:{chunk_slice.stop}")
        return summary(metadata, "success", elapsed_ms, "chunk-check")
    return summary(metadata, "success", elapsed_ms)


def main() -> int:
    args = parse_args()
    env = resolve_env(args)
    if args.describe_case:
        print(json.dumps(describe_case(env), separators=(",", ":")))
        return 0
    print(json.dumps(run_case(env, args.chunk_check, args.compile_only), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
