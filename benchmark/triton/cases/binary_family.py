#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from triton_case_common import describe_case, env_for_case, evaluate_shape, infer_broadcast_shape, parse_case, validate_case

try:
    import torch
except ImportError:
    torch = None

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


BINARY_RUNTIME_ASSERTIONS = 6
BINARY_EXECUTION_BUDGET = 32_000_000


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


def describe_bound_case(case_path: str | Path) -> dict[str, Any]:
    metadata = dict(describe_case(case_path))
    metadata["mode"] = "triton-binary-kernel"
    metadata["explicit_assertions"] += BINARY_RUNTIME_ASSERTIONS
    metadata["notes"] = merge_notes(metadata.get("notes", ""), "kernel=elemwise_add", "flattened-broadcast-add")
    return metadata


def run_bound_case(
    case_path: str | Path,
    *,
    device: str = "cuda",
    chunk_check: bool = False,
    compile_only: bool = False,
) -> dict[str, Any]:
    metadata = describe_bound_case(case_path)
    case = parse_case(case_path)
    if case.category != "elemwise_add":
        raise ValueError(f"{case.case_name}: expected elemwise_add, saw {case.category}")
    if not metadata["valid_contract"]:
        return summary(metadata, "n/a", 0.0, "Invalid contract")
    if triton is None or tl is None:
        return summary(metadata, "n/a", 0.0, "Missing triton package")
    if torch is None:
        return summary(metadata, "n/a", 0.0, "Missing torch package")
    if device != "cuda":
        return summary(metadata, "n/a", 0.0, "Binary Triton kernels require CUDA")
    if not torch.cuda.is_available():
        return summary(metadata, "n/a", 0.0, "CUDA not available")

    env = env_for_case(case)
    _, notes = validate_case(case, env)
    lhs_shape = evaluate_shape(case.inputs[0], env)
    rhs_is_scalar = case.inputs[1].is_scalar
    rhs_shape = () if rhs_is_scalar else evaluate_shape(case.inputs[1], env)
    output_shape = evaluate_shape(case.output, env)
    total_elements = math.prod(output_shape)
    if total_elements > BINARY_EXECUTION_BUDGET:
        return summary(metadata, "n/a", 0.0, f"Execution budget exceeded {total_elements} elements")

    lhs = torch.randn(lhs_shape, device="cuda", dtype=torch.float32).contiguous()
    if rhs_is_scalar:
        rhs = float(case.inputs[1].token)
        rhs_expanded = None
    else:
        rhs = torch.randn(rhs_shape, device="cuda", dtype=torch.float32).contiguous()
        rhs_expanded = expand_rhs(rhs, output_shape)
    lhs_expanded = lhs if lhs_shape == output_shape else expand_rhs(lhs, output_shape)
    block_size = choose_block_size(total_elements)
    runtime_assertions(lhs_expanded, rhs_expanded, rhs_is_scalar, block_size, case.case_name)

    started = time.perf_counter_ns()
    result = launch_add(lhs_expanded, rhs_expanded, rhs, output_shape, block_size)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0

    if compile_only:
        return summary(metadata, "success", elapsed_ms, *notes, f"block_size={block_size}", "compile-smoke")

    if chunk_check:
        reference = lhs_expanded + (rhs if rhs_is_scalar else rhs_expanded)
        validate_chunks(result, reference)
        notes.append("chunk-check")

    return summary(metadata, "success", elapsed_ms, *notes, f"block_size={block_size}")


def bound_case_main(case_path: str) -> int:
    parser = argparse.ArgumentParser(description="Run generated Triton binary add case")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--describe-case", action="store_true")
    parser.add_argument("--chunk-check", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    if args.describe_case:
        print(json.dumps(describe_bound_case(case_path), separators=(",", ":")))
        return 0

    print(json.dumps(run_bound_case(case_path, device=args.device, chunk_check=args.chunk_check, compile_only=args.compile_only), separators=(",", ":")))
    return 0


def expand_rhs(tensor: torch.Tensor, output_shape: tuple[int, ...]) -> torch.Tensor:
    prefix = (1,) * (len(output_shape) - tensor.ndim)
    reshaped = tensor.reshape(prefix + tuple(tensor.shape))
    return reshaped.expand(output_shape).contiguous()


def choose_block_size(numel: int) -> int:
    for candidate in (1024, 512, 256, 128, 64):
        if numel >= candidate:
            return candidate
    return 64


def runtime_assertions(
    lhs: torch.Tensor,
    rhs: torch.Tensor | None,
    rhs_is_scalar: bool,
    block_size: int,
    case_name: str,
) -> None:
    require(lhs.is_cuda, case_name, "lhs tensor must live on CUDA for Triton JIT")
    require(lhs.dtype == torch.float32, case_name, "lhs tensor must be float32")
    require(lhs.is_contiguous(), case_name, "lhs tensor must be contiguous after broadcast expansion")
    require(lhs.numel() > 0, case_name, "lhs tensor must have at least one element")
    if not rhs_is_scalar:
        require(rhs is not None and rhs.is_contiguous(), case_name, "rhs tensor must be contiguous after broadcast expansion")
    require(block_size in {64, 128, 256, 512, 1024}, case_name, "block size must be one of the supported intuition-selected tile sizes")


def require(condition: bool, case_name: str, message: str) -> None:
    if not condition:
        raise ValueError(f"{case_name}: {message}")


def launch_add(
    lhs: torch.Tensor,
    rhs_tensor: torch.Tensor | None,
    rhs_scalar: float | torch.Tensor,
    output_shape: tuple[int, ...],
    block_size: int,
) -> torch.Tensor:
    lhs_flat = lhs.reshape(-1)
    out_flat = torch.empty_like(lhs_flat)
    grid = (triton.cdiv(lhs_flat.numel(), block_size),)
    if rhs_tensor is None:
        add_scalar_kernel[grid](lhs_flat, float(rhs_scalar), out_flat, lhs_flat.numel(), block_size=block_size)
    else:
        add_tensor_kernel[grid](lhs_flat, rhs_tensor.reshape(-1), out_flat, lhs_flat.numel(), block_size=block_size)
    return out_flat.reshape(output_shape)


def validate_chunks(result: torch.Tensor, reference: torch.Tensor) -> None:
    flat_out = result.reshape(-1)
    flat_ref = reference.reshape(-1)
    chunk = min(4096, flat_out.numel())
    slices = [slice(0, chunk)]
    if flat_out.numel() > chunk:
        slices.append(slice(flat_out.numel() - chunk, flat_out.numel()))
    for chunk_slice in slices:
        if not torch.allclose(flat_out[chunk_slice], flat_ref[chunk_slice], atol=1e-4, rtol=1e-4):
            raise ValueError(f"chunk-check failed for elemwise_add at slice {chunk_slice.start}:{chunk_slice.stop}")


def merge_notes(*notes: str) -> str:
    ordered = []
    seen = set()
    for raw in notes:
        for piece in raw.split(";"):
            note = piece.strip()
            if not note or note in seen:
                continue
            seen.add(note)
            ordered.append(note)
    return "; ".join(ordered)


def summary(metadata: dict[str, Any], status: str, time_ms: float, *notes: str) -> dict[str, Any]:
    return {
        "status": status,
        "mode": metadata["mode"],
        "explicit_assertions": metadata["explicit_assertions"],
        "input_shape": metadata["input_shape"],
        "output_shape": metadata["output_shape"],
        "time_ms": round(time_ms, 4),
        "notes": merge_notes(metadata.get("notes", ""), *notes),
    }


if __name__ == "__main__":
    raise SystemExit("This module is intended to be imported by generated binary case entrypoints.")