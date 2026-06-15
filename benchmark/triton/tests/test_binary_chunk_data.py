#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CASES_DIR = ROOT / "benchmark" / "triton" / "cases"
if str(CASES_DIR) not in sys.path:
    sys.path.insert(0, str(CASES_DIR))

from binary_family import run_bound_case  # noqa: E402


DEFAULT_CASES = (
    "benchmark/choreo/elemwise_add/1_bert_32x512x768_32x512x768_32x512x768.co",
    "benchmark/choreo/elemwise_add/2_broadcast_128x256x28x28_256_128x256x28x28.co",
    "benchmark/choreo/elemwise_add/3_broadcast_32x512x768_1_32x512x768.co",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run chunk-data correctness tests for Triton binary-add kernels")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = [run_bound_case(ROOT / case_path, device=args.device, chunk_check=True) for case_path in DEFAULT_CASES]
    if args.json:
        print(json.dumps(results, separators=(",", ":")))
    else:
        for result in results:
            print(f"{result['status']} {result['mode']} {result['notes']}")
    return 1 if any(result["status"] != "success" for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())