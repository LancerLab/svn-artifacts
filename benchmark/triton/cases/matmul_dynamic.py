import argparse
import json
import time

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


EXPLICIT_ASSERTIONS = [
    "lhs.ndim == 2",
    "rhs.ndim == 2",
    "lhs.dtype == torch.float32",
    "rhs.dtype == torch.float32",
    "lhs.device.type == 'cuda'",
    "rhs.device.type == 'cuda'",
    "lhs.shape[0] > 0",
    "lhs.shape[1] == 2048",
    "rhs.shape[0] == 2048",
    "rhs.shape[1] == 1000",
]


if triton is not None:
    @triton.jit
    def matmul_dynamic_kernel(
        lhs_ptr,
        rhs_ptr,
        out_ptr,
        batch,
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

        for k_base in range(0, 2048, block_k):
            lhs_ptrs = lhs_ptr + offs_m[:, None] * lhs_stride_m + (k_base + offs_k[None, :]) * lhs_stride_k
            rhs_ptrs = rhs_ptr + (k_base + offs_k[:, None]) * rhs_stride_k + offs_n[None, :] * rhs_stride_n

            lhs = tl.load(lhs_ptrs, mask=offs_m[:, None] < batch, other=0.0)
            rhs = tl.load(rhs_ptrs, mask=offs_n[None, :] < 1000, other=0.0)
            acc += tl.dot(lhs, rhs)

        out_ptrs = out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n
        tl.store(out_ptrs, acc, mask=(offs_m[:, None] < batch) & (offs_n[None, :] < 1000))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_contract(lhs, rhs):
    require(lhs.ndim == 2, "lhs must be rank-2")
    require(rhs.ndim == 2, "rhs must be rank-2")
    require(lhs.dtype == torch.float32, "lhs must be float32")
    require(rhs.dtype == torch.float32, "rhs must be float32")
    require(lhs.device.type == "cuda", "lhs must be a CUDA tensor")
    require(rhs.device.type == "cuda", "rhs must be a CUDA tensor")
    require(lhs.shape[0] > 0, "batch dimension must be positive")
    require(lhs.shape[1] == 2048, "lhs second dimension must be 2048")
    require(rhs.shape[0] == 2048, "rhs first dimension must be 2048")
    require(rhs.shape[1] == 1000, "rhs second dimension must be 1000")


def matmul_dynamic(lhs, rhs):
    check_contract(lhs, rhs)
    batch = lhs.shape[0]
    out = torch.empty((batch, 1000), device=lhs.device, dtype=torch.float32)
    grid = (triton.cdiv(batch, 64), triton.cdiv(1000, 128))
    matmul_dynamic_kernel[grid](
        lhs,
        rhs,
        out,
        batch,
        lhs.stride(0),
        lhs.stride(1),
        rhs.stride(0),
        rhs.stride(1),
        out.stride(0),
        out.stride(1),
        block_m=64,
        block_n=128,
        block_k=32,
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    summary = {
        "status": "failed",
        "mode": "kernel",
        "explicit_assertions": len(EXPLICIT_ASSERTIONS),
        "input_shape": f"[{args.batch} x 2048] x [2048 x 1000]",
        "output_shape": f"[{args.batch} x 1000]",
        "notes": "",
        "time_ms": 0,
    }

    if torch is None or triton is None:
        summary["status"] = "n/a"
        summary["notes"] = "Missing torch or triton Python package"
        print(json.dumps(summary))
        raise SystemExit(2)

    if args.device != "cuda" or not torch.cuda.is_available():
        summary["status"] = "n/a"
        summary["notes"] = "CUDA runtime unavailable for Triton matmul case"
        print(json.dumps(summary))
        raise SystemExit(2)

    torch.manual_seed(args.seed)
    lhs = torch.randn((args.batch, 2048), device="cuda", dtype=torch.float32)
    rhs = torch.randn((2048, 1000), device="cuda", dtype=torch.float32)

    start = time.perf_counter_ns()
    out = matmul_dynamic(lhs, rhs)
    torch.cuda.synchronize()
    end = time.perf_counter_ns()

    ref = torch.matmul(lhs, rhs)
    max_abs_error = torch.max(torch.abs(out - ref)).item()
    summary["time_ms"] = (end - start) / 1_000_000.0
    summary["max_abs_error"] = max_abs_error

    if max_abs_error > args.tolerance:
        summary["status"] = "failed"
        summary["notes"] = f"max_abs_error={max_abs_error:.6f} exceeds tolerance"
        print(json.dumps(summary))
        raise SystemExit(1)

    summary["status"] = "success"
    summary["notes"] = "Operator-view shape contract validated against torch.matmul"
    print(json.dumps(summary))


if __name__ == "__main__":
    main()