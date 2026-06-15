#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CASE_ROOT = WORKSPACE_ROOT / "benchmark" / "choreo"
GENERATED_ROOT = WORKSPACE_ROOT / "benchmark" / "triton"

sys.path.insert(0, str((WORKSPACE_ROOT / "benchmark" / "triton" / "cases").resolve()))

from triton_case_common import (  # noqa: E402
    apply_reduction_shape,
    describe_case,
    env_for_case,
    evaluate_shape,
    infer_batch_norm_axis,
    infer_concat_axis,
    infer_conv2d_params,
    infer_permutation,
    infer_pool2d_params,
    infer_reduction,
    parse_case,
)


UNARY_CATEGORIES = {"relu", "sigmoid", "gelu"}
REAL_KERNEL_CATEGORIES = {"relu", "sigmoid", "gelu", "elemwise_add", "matmul"}


def main() -> int:
    choreo_cases = sorted(path for path in CASE_ROOT.rglob("*.co") if path.stem[:1].isdigit())
    for case_path in choreo_cases:
        relative_case = case_path.relative_to(WORKSPACE_ROOT).as_posix()
        case = parse_case(case_path)
        target = GENERATED_ROOT / case.category / f"{case.case_name}.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_case(relative_case), encoding="utf-8")
        target.chmod(0o755)
    print(f"Rendered {len(choreo_cases)} standalone Triton case files under {GENERATED_ROOT}/<category>/")
    return 0


def render_case(relative_case: str) -> str:
    case_path = WORKSPACE_ROOT / relative_case
    case = parse_case(case_path)
    metadata = describe_case(case_path)
    env = env_for_case(case)
    symbols = sorted({symbol for value in case.inputs + (case.output,) if not value.is_scalar for dim in value.dims for symbol in _symbols_in_dim(dim)})
    symbol_defaults = {symbol: env[symbol] for symbol in symbols}

    context = {
        "case_path": relative_case,
        "case_name": case.case_name,
        "category": case.category,
        "symbols": symbols,
        "symbol_defaults": symbol_defaults,
        "input_tokens": [value.token for value in case.inputs],
        "output_token": case.output.token,
        "notes": metadata.get("notes", ""),
    }

    if case.category in UNARY_CATEGORIES:
        return render_unary_case(context)
    if case.category == "elemwise_add":
        return render_elemwise_add_case(context)
    if case.category == "matmul":
        return render_matmul_case(context)
    return render_host_standalone_case(context)


def render_common_header(context: dict[str, object]) -> str:
    parser_args = []
    for symbol in context["symbols"]:
        default = context["symbol_defaults"][symbol]
        parser_args.append(f'parser.add_argument("--{symbol}", type=int, default={default})')
    parser_args_text = "\n".join(parser_args)
    parser_args_block = f"{parser_args_text}\n" if parser_args_text else ""

    header = textwrap.dedent(
        f"""\
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


        CASE_PATH = {context["case_path"]!r}
        CASE_NAME = {context["case_name"]!r}
        CATEGORY = {context["category"]!r}
        SYMBOL_DEFAULTS = {context["symbol_defaults"]!r}
        INPUT_TOKENS = {context["input_tokens"]!r}
        OUTPUT_TOKEN = {context["output_token"]!r}
        BASE_NOTES = {context["notes"]!r}
        ATOM_RE = re.compile(r"[A-Z][0-9](?![0-9])|[A-Z]|\\d+")


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
                raise ValueError(f"{{CASE_NAME}}: {{message}}")


        def _parse_atom(expr: str, index: int) -> tuple[str, int]:
            match = ATOM_RE.match(expr, index)
            if not match:
                raise ValueError(f"invalid dimension expression: {{expr}}")
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
            return {{
                "status": status,
                "mode": metadata["mode"],
                "explicit_assertions": metadata["explicit_assertions"],
                "input_shape": metadata["input_shape"],
                "output_shape": metadata["output_shape"],
                "time_ms": round(time_ms, 4),
                "notes": merge_notes(metadata.get("notes", ""), *notes),
            }}


        def parse_args() -> argparse.Namespace:
            parser = argparse.ArgumentParser(description=f"Standalone Triton case for {{CATEGORY}}")
        __PARSER_ARGS__
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
        """
    )
    return header.replace("__PARSER_ARGS__\n", textwrap.indent(parser_args_block, "    "))


def render_unary_case(context: dict[str, object]) -> str:
    op_name = context["category"]
    op_body = {
        "relu": "tl.maximum(values, 0.0)",
        "sigmoid": "tl.sigmoid(values)",
        # tanh(t) = 2*sigmoid(2*t) - 1  (avoids tl.math.tanh removed in Triton 3.x)
        "gelu": "values * tl.sigmoid(2.0 * 0.7978845608028654 * (values + 0.044715 * values * values * values))",
    }[op_name]
    return render_common_header(context) + textwrap.dedent(
        f"""\

        RUNTIME_ASSERTIONS = 7
        BLOCK_SIZE = 1024


        if triton is not None and tl is not None:

            @triton.jit
            def op_kernel(x_ptr, y_ptr, numel, block_size: tl.constexpr):
                pid = tl.program_id(axis=0)
                offsets = pid * block_size + tl.arange(0, block_size)
                mask = offsets < numel
                values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                out = {op_body}
                tl.store(y_ptr + offsets, out, mask=mask)


        def describe_case(env: dict[str, int]) -> dict[str, object]:
            shape = eval_shape(INPUT_TOKENS[0], env)
            return {{
                "case_name": CASE_NAME,
                "category": CATEGORY,
                "choreo_case": CASE_PATH,
                "mode": "triton-standalone-kernel",
                "input_shape": str(shape),
                "output_shape": str(shape),
                "explicit_assertions": RUNTIME_ASSERTIONS,
                "notes": merge_notes(BASE_NOTES, "standalone unary kernel", f"op={op_name}", f"block_size={{BLOCK_SIZE}}"),
            }}


        def runtime_assertions(tensor) -> None:
            # EXPLICIT_ASSERTION: tensor.ndim >= 1
            require(tensor.ndim >= 1, "input rank must be positive")
            # EXPLICIT_ASSERTION: tensor.is_cuda
            require(tensor.is_cuda, "tensor must live on CUDA for Triton JIT")
            # EXPLICIT_ASSERTION: tensor.dtype == torch.float32
            require(tensor.dtype == torch.float32, "tensor must be float32")
            # EXPLICIT_ASSERTION: tensor.is_contiguous()
            require(tensor.is_contiguous(), "tensor must be contiguous")
            # EXPLICIT_ASSERTION: tensor.numel() > 0
            require(tensor.numel() > 0, "tensor must have at least one element")
            # EXPLICIT_ASSERTION: BLOCK_SIZE in {{64, 128, 256, 512, 1024}}
            require(BLOCK_SIZE in {{64, 128, 256, 512, 1024}}, "block size must be supported")
            # EXPLICIT_ASSERTION: tensor.numel() >= 1
            require(tensor.numel() >= 1, "tensor element count must be positive")


        def reference_impl(tensor):
            if CATEGORY == "relu":
                return torch.relu(tensor)
            if CATEGORY == "sigmoid":
                return torch.sigmoid(tensor)
            return torch_f.gelu(tensor, approximate="tanh")


        def run_case(env: dict[str, int], chunk_check: bool, compile_only: bool) -> dict[str, object]:
            metadata = describe_case(env)
            if triton is None or tl is None:
                return summary(metadata, "n/a", 0.0, "Missing triton package")
            if torch is None or torch_f is None:
                return summary(metadata, "n/a", 0.0, "Missing torch package")
            if not torch.cuda.is_available():
                return summary(metadata, "n/a", 0.0, "CUDA not available")

            shape = eval_shape(INPUT_TOKENS[0], env)
            tensor = torch.randn(shape, device="cuda", dtype=torch.float32).contiguous()
            runtime_assertions(tensor)
            flat = tensor.reshape(-1)
            out = torch.empty_like(flat)
            started = time.perf_counter_ns()
            grid = (triton.cdiv(flat.numel(), BLOCK_SIZE),)
            op_kernel[grid](flat, out, flat.numel(), block_size=BLOCK_SIZE)
            result = out.reshape(tensor.shape)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            if compile_only:
                return summary(metadata, "success", elapsed_ms, "compile-smoke")
            if chunk_check:
                reference = reference_impl(tensor)
                chunk = min(4096, result.numel())
                slices = [slice(0, chunk)]
                if result.numel() > chunk:
                    slices.append(slice(result.numel() - chunk, result.numel()))
                flat_result = result.reshape(-1)
                flat_reference = reference.reshape(-1)
                for chunk_slice in slices:
                    if not torch.allclose(flat_result[chunk_slice], flat_reference[chunk_slice], atol=1e-4, rtol=1e-4):
                        raise ValueError(f"{{CASE_NAME}}: chunk validation failed at slice {{chunk_slice.start}}:{{chunk_slice.stop}}")
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
        """
    )


def render_elemwise_add_case(context: dict[str, object]) -> str:
    rhs_scalar = len(context["input_tokens"]) > 1 and str(context["input_tokens"][1]).isdigit()
    return render_common_header(context) + textwrap.dedent(
        f"""\

        RUNTIME_ASSERTIONS = 8
        BLOCK_SIZE = 1024
        RHS_SCALAR = {rhs_scalar!r}


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
            return {{
                "case_name": CASE_NAME,
                "category": CATEGORY,
                "choreo_case": CASE_PATH,
                "mode": "triton-standalone-kernel",
                "input_shape": f"{{lhs_shape}} + {{rhs_shape}}",
                "output_shape": str(output_shape),
                "explicit_assertions": RUNTIME_ASSERTIONS,
                "notes": merge_notes(BASE_NOTES, "standalone binary add kernel", f"block_size={{BLOCK_SIZE}}"),
            }}


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
            # EXPLICIT_ASSERTION: BLOCK_SIZE in {{64, 128, 256, 512, 1024}}
            require(BLOCK_SIZE in {{64, 128, 256, 512, 1024}}, "block size must be supported")


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
                        raise ValueError(f"{{CASE_NAME}}: chunk validation failed at slice {{chunk_slice.start}}:{{chunk_slice.stop}}")
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
        """
    )


def render_matmul_case(context: dict[str, object]) -> str:
    return render_common_header(context) + textwrap.dedent(
        """\

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
        """
    )


def render_host_standalone_case(context: dict[str, object]) -> str:
    case_path = WORKSPACE_ROOT / str(context["case_path"])
    case = parse_case(case_path)
    env = env_for_case(case)
    input_shapes = [evaluate_shape(value, env) for value in case.inputs if not value.is_scalar]
    output_shape = evaluate_shape(case.output, env)

    extras = {}
    if case.category == "transpose":
        extras["permutation"] = infer_permutation(case.inputs[0].dims, case.output.dims)
    elif case.category == "concat":
        extras["axis"] = infer_concat_axis(input_shapes, output_shape)
    elif case.category == "reduce_mean":
        extras["reduction"] = infer_reduction(case.inputs[0].dims, case.output.dims)
    elif case.category == "batch_norm":
        extras["param_axis"] = infer_batch_norm_axis(input_shapes[0], input_shapes[1])
    elif case.category == "conv2d":
        stride, padding, dilation = infer_conv2d_params(case, input_shapes[0], input_shapes[1], output_shape)
        extras.update({"stride": stride, "padding": padding, "dilation": dilation})
    elif case.category == "max_pool2d":
        kernel, stride, padding = infer_pool2d_params(input_shapes[0], output_shape)
        extras.update({"kernel": kernel, "stride": stride, "padding": padding})

    input_scalar_flags = tuple(value.is_scalar for value in case.inputs)

    return render_common_header(context) + textwrap.dedent(
        f"""\

        RUNTIME_ASSERTIONS = 6
        HOST_EXTRAS = {extras!r}
        INPUT_IS_SCALAR = {input_scalar_flags!r}


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
            mode = "triton-standalone-kernel" if CATEGORY in {REAL_KERNEL_CATEGORIES!r} else "triton-standalone-host-entrypoint"
            return {{
                "case_name": CASE_NAME,
                "category": CATEGORY,
                "choreo_case": CASE_PATH,
                "mode": mode,
                "input_shape": " + ".join(input_desc),
                "output_shape": str(output_shape),
                "explicit_assertions": RUNTIME_ASSERTIONS,
                "notes": merge_notes(BASE_NOTES, "standalone host implementation", "triton copy epilogue"),
            }}


        def runtime_assertions(*tensors) -> None:
            present = [tensor for tensor in tensors if isinstance(tensor, torch.Tensor)]
            # EXPLICIT_ASSERTION: len(present) >= 1
            require(len(present) >= 1, "at least one tensor input is required")
            # EXPLICIT_ASSERTION: all(tensor.is_cuda for tensor in present)
            require(all(tensor.is_cuda for tensor in present), "all tensors must live on CUDA")
            # EXPLICIT_ASSERTION: all(tensor.dtype in {{torch.float32, torch.int64}} for tensor in present)
            require(all(tensor.dtype in {{torch.float32, torch.int64}} for tensor in present), "tensor dtypes must be supported")
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

            raise ValueError(f"{{CASE_NAME}}: unsupported standalone host category {{CATEGORY}}")


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
                        raise ValueError(f"{{CASE_NAME}}: chunk validation failed at slice {{chunk_slice.start}}:{{chunk_slice.stop}}")
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
        """
    )


def _symbols_in_dim(dim: str) -> list[str]:
    symbols: list[str] = []
    token = ""
    for char in dim:
        if char.isalnum():
            token += char
            continue
        if token and token[0].isalpha():
            symbols.append(token)
        token = ""
    if token and token[0].isalpha():
        symbols.append(token)
    return symbols


if __name__ == "__main__":
    raise SystemExit(main())