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


CASE_PATH = 'benchmark/choreo/matmul/9_dynamic_16xSx512_512x256_16xSx256.co'
CASE_NAME = '9_dynamic_16xSx512_512x256_16xSx256'
CATEGORY = 'matmul'
SYMBOL_DEFAULTS = {'S': 20}
INPUT_TOKENS = ['16xSx512', '512x256']
OUTPUT_TOKEN = '16xSx256'
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
    parser.add_argument("--S", type=int, default=20)
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

RUNTIME_ASSERTIONS = 5
BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 32


if triton is not None and tl is not None:

    @triton.jit
    def matmul_kernel(
        lhs_ptr,
        rhs_ptr,
        out_ptr,
        m,
        n,
        k,
        lhs_stride_m,
        lhs_stride_k,
        rhs_stride_k,
        rhs_stride_n,
        out_stride_m,
        out_stride_n,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        offs_m = pid_m * block_m + tl.arange(0, block_m)
        offs_n = pid_n * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)
        acc = tl.zeros((block_m, block_n), dtype=tl.float32)
        for k_start in range(0, k, block_k):
            lhs_ptrs = lhs_ptr + offs_m[:, None] * lhs_stride_m + (k_start + offs_k)[None, :] * lhs_stride_k
            rhs_ptrs = rhs_ptr + (k_start + offs_k)[:, None] * rhs_stride_k + offs_n[None, :] * rhs_stride_n
            lhs_mask = (offs_m[:, None] < m) & ((k_start + offs_k)[None, :] < k)
            rhs_mask = ((k_start + offs_k)[:, None] < k) & (offs_n[None, :] < n)
            lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
            rhs = tl.load(rhs_ptrs, mask=rhs_mask, other=0.0)
            acc += tl.dot(lhs, rhs)
        out_ptrs = out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n
        out_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
        tl.store(out_ptrs, acc, mask=out_mask)


def broadcast_batch_shape(lhs_batch: tuple[int, ...], rhs_batch: tuple[int, ...]) -> tuple[int, ...]:
    return torch.broadcast_shapes(lhs_batch, rhs_batch)


def describe_case(env: dict[str, int]) -> dict[str, object]:
    lhs_shape = eval_shape(INPUT_TOKENS[0], env)
    rhs_shape = eval_shape(INPUT_TOKENS[1], env)
    output_shape = eval_shape(OUTPUT_TOKEN, env)
    return {
        "case_name": CASE_NAME,
        "category": CATEGORY,
        "choreo_case": CASE_PATH,
        "mode": "triton-standalone-kernel",
        "input_shape": f"{lhs_shape} + {rhs_shape}",
        "output_shape": str(output_shape),
        "explicit_assertions": RUNTIME_ASSERTIONS,
        "notes": merge_notes(BASE_NOTES, "standalone matmul kernel", f"tiles={BLOCK_M}x{BLOCK_N}x{BLOCK_K}"),
    }


def runtime_assertions(lhs, rhs) -> None:
    # EXPLICIT_ASSERTION: lhs.ndim >= 2 and rhs.ndim >= 2
    require(lhs.ndim >= 2 and rhs.ndim >= 2, "matmul operands must be rank >= 2")
    # EXPLICIT_ASSERTION: lhs.shape[-1] == rhs.shape[-2]
    require(lhs.shape[-1] == rhs.shape[-2], "inner dimensions must match")
    # EXPLICIT_ASSERTION: lhs.is_cuda and rhs.is_cuda
    require(lhs.is_cuda and rhs.is_cuda, "operands must live on CUDA")
    # EXPLICIT_ASSERTION: lhs.dtype == torch.float32 and rhs.dtype == torch.float32
    require(lhs.dtype == torch.float32 and rhs.dtype == torch.float32, "operands must be float32")
    # EXPLICIT_ASSERTION: lhs.is_contiguous() and rhs.is_contiguous()
    require(lhs.is_contiguous() and rhs.is_contiguous(), "operands must be contiguous")


def run_case(env: dict[str, int], chunk_check: bool, compile_only: bool) -> dict[str, object]:
    metadata = describe_case(env)
    if triton is None or tl is None:
        return summary(metadata, "n/a", 0.0, "Missing triton package")
    if torch is None:
        return summary(metadata, "n/a", 0.0, "Missing torch package")
    if not torch.cuda.is_available():
        return summary(metadata, "n/a", 0.0, "CUDA not available")

    lhs_shape = eval_shape(INPUT_TOKENS[0], env)
    rhs_shape = eval_shape(INPUT_TOKENS[1], env)
    lhs = torch.randn(lhs_shape, device="cuda", dtype=torch.float32).contiguous()
    rhs = torch.randn(rhs_shape, device="cuda", dtype=torch.float32).contiguous()
    runtime_assertions(lhs, rhs)

    batch_shape = broadcast_batch_shape(tuple(lhs.shape[:-2]), tuple(rhs.shape[:-2]))
    lhs_batch = lhs.expand(batch_shape + lhs.shape[-2:]).contiguous().reshape(-1, lhs.shape[-2], lhs.shape[-1])
    rhs_batch = rhs.expand(batch_shape + rhs.shape[-2:]).contiguous().reshape(-1, rhs.shape[-2], rhs.shape[-1])
    out_batch = torch.empty((lhs_batch.shape[0], lhs.shape[-2], rhs.shape[-1]), device="cuda", dtype=torch.float32)

    started = time.perf_counter_ns()
    for batch_index in range(lhs_batch.shape[0]):
        lhs_tile = lhs_batch[batch_index]
        rhs_tile = rhs_batch[batch_index]
        out_tile = out_batch[batch_index]
        grid = (triton.cdiv(lhs_tile.shape[0], BLOCK_M), triton.cdiv(rhs_tile.shape[1], BLOCK_N))
        matmul_kernel[grid](
            lhs_tile,
            rhs_tile,
            out_tile,
            lhs_tile.shape[0],
            rhs_tile.shape[1],
            lhs_tile.shape[1],
            lhs_tile.stride(0),
            lhs_tile.stride(1),
            rhs_tile.stride(0),
            rhs_tile.stride(1),
            out_tile.stride(0),
            out_tile.stride(1),
            block_m=BLOCK_M,
            block_n=BLOCK_N,
            block_k=BLOCK_K,
        )
    result = out_batch.reshape(batch_shape + (lhs.shape[-2], rhs.shape[-1]))
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    if compile_only:
        return summary(metadata, "success", elapsed_ms, "compile-smoke")
    if chunk_check:
        reference = torch.matmul(lhs, rhs)
        flat_result = result.reshape(-1)
        flat_reference = reference.reshape(-1)
        chunk = min(4096, flat_result.numel())
        slices = [slice(0, chunk)]
        if flat_result.numel() > chunk:
            slices.append(slice(flat_result.numel() - chunk, flat_result.numel()))
        for chunk_slice in slices:
            if not torch.allclose(flat_result[chunk_slice], flat_reference[chunk_slice], atol=1e-3, rtol=1e-3):
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
