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


CASE_PATH = 'benchmark/choreo/elemwise_add/19_transformer_32x512x2048_32x512x2048_32x512x2048.co'
CASE_NAME = '19_transformer_32x512x2048_32x512x2048_32x512x2048'
CATEGORY = 'elemwise_add'
SYMBOL_DEFAULTS = {}
INPUT_TOKENS = ['32x512x2048', '32x512x2048']
OUTPUT_TOKEN = '32x512x2048'
BASE_NOTES = ''
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

RUNTIME_ASSERTIONS = 8
BLOCK_SIZE = 1024
RHS_SCALAR = False


if triton is not None and tl is not None:

    @triton.jit
    def add_tensor_kernel(lhs_ptr, rhs_ptr, out_ptr, numel, block_size: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < numel
        lhs = tl.load(lhs_ptr + offsets, mask=mask, other=0.0)
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, lhs + rhs, mask=mask)


    @triton.jit
    def add_scalar_kernel(lhs_ptr, scalar, out_ptr, numel, block_size: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < numel
        lhs = tl.load(lhs_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, lhs + scalar, mask=mask)


def expand_to_shape(tensor, output_shape):
    prefix = (1,) * (len(output_shape) - tensor.ndim)
    return tensor.reshape(prefix + tuple(tensor.shape)).expand(output_shape).contiguous()


def describe_case(env: dict[str, int]) -> dict[str, object]:
    lhs_shape = eval_shape(INPUT_TOKENS[0], env)
    output_shape = eval_shape(OUTPUT_TOKEN, env)
    rhs_shape = "scalar" if RHS_SCALAR else str(eval_shape(INPUT_TOKENS[1], env))
    return {
        "case_name": CASE_NAME,
        "category": CATEGORY,
        "choreo_case": CASE_PATH,
        "mode": "triton-standalone-kernel",
        "input_shape": f"{lhs_shape} + {rhs_shape}",
        "output_shape": str(output_shape),
        "explicit_assertions": RUNTIME_ASSERTIONS,
        "notes": merge_notes(BASE_NOTES, "standalone binary add kernel", f"block_size={BLOCK_SIZE}"),
    }


def runtime_assertions(lhs, rhs, rhs_is_scalar: bool) -> None:
    # EXPLICIT_ASSERTION: lhs.ndim >= 1
    require(lhs.ndim >= 1, "lhs rank must be positive")
    # EXPLICIT_ASSERTION: lhs.is_cuda
    require(lhs.is_cuda, "lhs must live on CUDA")
    # EXPLICIT_ASSERTION: lhs.dtype == torch.float32
    require(lhs.dtype == torch.float32, "lhs must be float32")
    # EXPLICIT_ASSERTION: lhs.is_contiguous()
    require(lhs.is_contiguous(), "lhs must be contiguous")
    # EXPLICIT_ASSERTION: lhs.numel() > 0
    require(lhs.numel() > 0, "lhs must have positive element count")
    # EXPLICIT_ASSERTION: rhs_is_scalar or rhs is not None
    require(rhs_is_scalar or rhs is not None, "rhs tensor must exist for tensor add")
    # EXPLICIT_ASSERTION: rhs_is_scalar or rhs.is_contiguous()
    require(rhs_is_scalar or rhs.is_contiguous(), "rhs tensor must be contiguous after expansion")
    # EXPLICIT_ASSERTION: BLOCK_SIZE in {64, 128, 256, 512, 1024}
    require(BLOCK_SIZE in {64, 128, 256, 512, 1024}, "block size must be supported")


def run_case(env: dict[str, int], chunk_check: bool, compile_only: bool) -> dict[str, object]:
    metadata = describe_case(env)
    if triton is None or tl is None:
        return summary(metadata, "n/a", 0.0, "Missing triton package")
    if torch is None:
        return summary(metadata, "n/a", 0.0, "Missing torch package")
    if not torch.cuda.is_available():
        return summary(metadata, "n/a", 0.0, "CUDA not available")

    lhs_shape = eval_shape(INPUT_TOKENS[0], env)
    output_shape = eval_shape(OUTPUT_TOKEN, env)
    lhs = torch.randn(lhs_shape, device="cuda", dtype=torch.float32).contiguous()
    rhs_tensor = None
    rhs_value = None
    if RHS_SCALAR:
        rhs_value = float(INPUT_TOKENS[1])
        lhs_expanded = expand_to_shape(lhs, output_shape) if lhs.shape != output_shape else lhs
    else:
        rhs_shape = eval_shape(INPUT_TOKENS[1], env)
        rhs_tensor = torch.randn(rhs_shape, device="cuda", dtype=torch.float32).contiguous()
        lhs_expanded = expand_to_shape(lhs, output_shape) if lhs.shape != output_shape else lhs
        rhs_tensor = expand_to_shape(rhs_tensor, output_shape) if rhs_tensor.shape != output_shape else rhs_tensor
    runtime_assertions(lhs_expanded, rhs_tensor, RHS_SCALAR)
    lhs_flat = lhs_expanded.reshape(-1)
    out_flat = torch.empty_like(lhs_flat)
    started = time.perf_counter_ns()
    grid = (triton.cdiv(lhs_flat.numel(), BLOCK_SIZE),)
    if RHS_SCALAR:
        add_scalar_kernel[grid](lhs_flat, rhs_value, out_flat, lhs_flat.numel(), block_size=BLOCK_SIZE)
        reference = lhs_expanded + rhs_value
    else:
        add_tensor_kernel[grid](lhs_flat, rhs_tensor.reshape(-1), out_flat, lhs_flat.numel(), block_size=BLOCK_SIZE)
        reference = lhs_expanded + rhs_tensor
    result = out_flat.reshape(output_shape)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    if compile_only:
        return summary(metadata, "success", elapsed_ms, "compile-smoke")
    if chunk_check:
        chunk = min(4096, result.numel())
        slices = [slice(0, chunk)]
        if result.numel() > chunk:
            slices.append(slice(result.numel() - chunk, result.numel()))
        flat_result = result.reshape(-1)
        flat_reference = reference.reshape(-1)
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
