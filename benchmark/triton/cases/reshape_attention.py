import argparse
import json
import time

try:
    import torch
except ImportError:
    torch = None


EXPLICIT_ASSERTIONS = [
    "input.ndim == 3",
    "input.dtype == torch.float32",
    "input.shape[0] > 0",
    "input.shape[1] > 0",
    "input.shape[2] == 768",
    "12 > 0",
    "64 > 0",
    "12 * 64 == 768",
]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def reshape_attention(tensor):
    require(tensor.ndim == 3, "input must be rank-3")
    require(tensor.dtype == torch.float32, "input must be float32")
    require(tensor.shape[0] > 0, "batch dimension must be positive")
    require(tensor.shape[1] > 0, "sequence dimension must be positive")
    require(tensor.shape[2] == 768, "last dimension must be 768")
    require(12 > 0, "head count must be positive")
    require(64 > 0, "head width must be positive")
    require(12 * 64 == 768, "attention reshape requires 768 == 12 * 64")
    return tensor.reshape(tensor.shape[0], tensor.shape[1], 12, 64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary = {
        "status": "failed",
        "mode": "host",
        "explicit_assertions": len(EXPLICIT_ASSERTIONS),
        "input_shape": f"[{args.batch} x {args.seq_len} x 768]",
        "output_shape": f"[{args.batch} x {args.seq_len} x 12 x 64]",
        "notes": "",
        "time_ms": 0,
    }

    if torch is None:
        summary["status"] = "n/a"
        summary["notes"] = "Missing torch Python package"
        print(json.dumps(summary))
        raise SystemExit(2)

    if args.device == "cuda" and not torch.cuda.is_available():
        summary["status"] = "n/a"
        summary["notes"] = "CUDA runtime unavailable for requested device"
        print(json.dumps(summary))
        raise SystemExit(2)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    tensor = torch.randn((args.batch, args.seq_len, 768), device=device, dtype=torch.float32)

    start = time.perf_counter_ns()
    out = reshape_attention(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter_ns()

    summary["time_ms"] = (end - start) / 1_000_000.0
    summary["same_storage"] = out.storage().data_ptr() == tensor.storage().data_ptr()
    summary["status"] = "success"
    summary["notes"] = "Operator-view shape contract validated for reshape"
    print(json.dumps(summary))


if __name__ == "__main__":
    main()