#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from triton_case_common import describe_case, env_for_case, evaluate_shape, parse_case, validate_case

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


UNARY_RUNTIME_ASSERTIONS = 5
UNARY_EXECUTION_BUDGET = 16_000_000
SUPPORTED_UNARY_CATEGORIES = {"relu", "sigmoid", "gelu"}


if triton is not None and tl is not None:

    @triton.jit
    def relu_kernel(x_ptr, y_ptr, numel, block_size: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < numel
        values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(y_ptr + offsets, tl.maximum(values, 0.0), mask=mask)


    @triton.jit
    def sigmoid_kernel(x_ptr, y_ptr, numel, block_size: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < numel
        values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(y_ptr + offsets, tl.sigmoid(values), mask=mask)


    @triton.jit
    def gelu_kernel(x_ptr, y_ptr, numel, block_size: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < numel
        values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        coeff = 0.7978845608028654
        inner = coeff * (values + 0.044715 * values * values * values)
        # tanh(t) = 2*sigmoid(2*t) - 1  (avoids tl.math.tanh removed in Triton 3.x)
        gelu = values * tl.sigmoid(2.0 * inner)
        tl.store(y_ptr + offsets, gelu, mask=mask)


def describe_bound_case(case_path: str | Path, op_name: str) -> dict[str, Any]:
    metadata = dict(describe_case(case_path))
    metadata["mode"] = "triton-unary-kernel"
    metadata["explicit_assertions"] += UNARY_RUNTIME_ASSERTIONS
    metadata["notes"] = merge_notes(
        metadata.get("notes", ""),
        f"kernel={op_name}",
        "flattened-1d-tiling",
    )
    return metadata


def run_bound_case(
    case_path: str | Path,
    op_name: str,
    *,
    device: str = "cuda",
    chunk_check: bool = False,
    compile_only: bool = False,
) -> dict[str, Any]:
    metadata = describe_bound_case(case_path, op_name)
    case = parse_case(case_path)
    if case.category != op_name:
        raise ValueError(f"{case.case_name}: expected category {op_name}, saw {case.category}")
    if not metadata["valid_contract"]:
        return summary(metadata, "n/a", 0.0, "Invalid contract")
    if case.category not in SUPPORTED_UNARY_CATEGORIES:
        raise ValueError(f"Unsupported unary category: {case.category}")
    if triton is None or tl is None:
        return summary(metadata, "n/a", 0.0, "Missing triton package")
    if torch is None or torch_f is None:
        return summary(metadata, "n/a", 0.0, "Missing torch package")
    if device != "cuda":
        return summary(metadata, "n/a", 0.0, "Unary Triton kernels require CUDA")
    if not torch.cuda.is_available():
        return summary(metadata, "n/a", 0.0, "CUDA not available")

    env = env_for_case(case)
    _, notes = validate_case(case, env)
    input_shape = evaluate_shape(case.inputs[0], env)
    if math.prod(input_shape) > UNARY_EXECUTION_BUDGET:
        return summary(metadata, "n/a", 0.0, f"Execution budget exceeded {math.prod(input_shape)} elements")

    tensor = torch.randn(input_shape, device="cuda", dtype=torch.float32)
    block_size = choose_block_size(tensor.numel())
    runtime_assertions(tensor, block_size, case.case_name)

    started = time.perf_counter_ns()
    result = launch_kernel(op_name, tensor, block_size)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0

    if compile_only:
        return summary(metadata, "success", elapsed_ms, *notes, f"block_size={block_size}", "compile-smoke")

    if chunk_check:
        validate_chunk_data(op_name, tensor, result)
        notes.append("chunk-check")

    return summary(metadata, "success", elapsed_ms, *notes, f"block_size={block_size}")


def bound_case_main(case_path: str, op_name: str) -> int:
    parser = argparse.ArgumentParser(description=f"Run generated Triton unary case for {op_name}")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--describe-case", action="store_true")
    parser.add_argument("--chunk-check", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    if args.describe_case:
        print(json.dumps(describe_bound_case(case_path, op_name), separators=(",", ":")))
        return 0

    print(
        json.dumps(
            run_bound_case(case_path, op_name, device=args.device, chunk_check=args.chunk_check, compile_only=args.compile_only),
            separators=(",", ":"),
        )
    )
    return 0


def choose_block_size(numel: int) -> int:
    for candidate in (1024, 512, 256, 128, 64):
        if numel >= candidate:
            return candidate
    return 64


def runtime_assertions(tensor: torch.Tensor, block_size: int, case_name: str) -> None:
    require(tensor.is_cuda, case_name, "input tensor must live on CUDA for Triton JIT")
    require(tensor.dtype == torch.float32, case_name, "input tensor must be float32")
    require(tensor.is_contiguous(), case_name, "input tensor must be contiguous")
    require(tensor.numel() > 0, case_name, "input tensor must have at least one element")
    require(block_size in {64, 128, 256, 512, 1024}, case_name, "block size must be one of the supported intuition-selected tile sizes")


def require(condition: bool, case_name: str, message: str) -> None:
    if not condition:
        raise ValueError(f"{case_name}: {message}")


def launch_kernel(op_name: str, tensor: torch.Tensor, block_size: int) -> torch.Tensor:
    flat = tensor.reshape(-1)
    output = torch.empty_like(flat)
    grid = (triton.cdiv(flat.numel(), block_size),)

    if op_name == "relu":
        relu_kernel[grid](flat, output, flat.numel(), block_size=block_size)
    elif op_name == "sigmoid":
        sigmoid_kernel[grid](flat, output, flat.numel(), block_size=block_size)
    elif op_name == "gelu":
        gelu_kernel[grid](flat, output, flat.numel(), block_size=block_size)
    else:
        raise ValueError(f"Unsupported unary op {op_name}")

    return output.reshape(tensor.shape)


def validate_chunk_data(op_name: str, tensor: torch.Tensor, result: torch.Tensor) -> None:
    flat_in = tensor.reshape(-1)
    flat_out = result.reshape(-1)
    flat_ref = reference_impl(op_name, flat_in).reshape(-1)
    chunk = min(4096, flat_in.numel())
    slices = [slice(0, chunk)]
    if flat_in.numel() > chunk:
        slices.append(slice(flat_in.numel() - chunk, flat_in.numel()))
    for chunk_slice in slices:
        if not torch.allclose(flat_out[chunk_slice], flat_ref[chunk_slice], atol=1e-4, rtol=1e-4):
            raise ValueError(f"chunk-check failed for {op_name} at slice {chunk_slice.start}:{chunk_slice.stop}")


def reference_impl(op_name: str, tensor: torch.Tensor) -> torch.Tensor:
    if op_name == "relu":
        return torch.relu(tensor)
    if op_name == "sigmoid":
        return torch.sigmoid(tensor)
    if op_name == "gelu":
        return torch_f.gelu(tensor, approximate="tanh")
    raise ValueError(f"Unsupported unary op {op_name}")


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
    raise SystemExit("This module is intended to be imported by generated unary case entrypoints.")