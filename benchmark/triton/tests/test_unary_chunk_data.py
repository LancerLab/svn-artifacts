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

from unary_family import run_bound_case  # noqa: E402


DEFAULT_CASES = (
    ("relu", "benchmark/choreo/relu/1_bert_32x512x768_32x512x768.co"),
    ("sigmoid", "benchmark/choreo/sigmoid/1_bert_32x512x768.co"),
    ("gelu", "benchmark/choreo/gelu/1_bert_32x512x768.co"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run chunk-data correctness tests for Triton unary kernels")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = []
    for category, case_path in DEFAULT_CASES:
        results.append(run_bound_case(ROOT / case_path, category, device=args.device, chunk_check=True))

    if args.json:
        print(json.dumps(results, separators=(",", ":")))
    else:
        for result in results:
            print(f"{result['status']} {result['mode']} {result['notes']}")

    failing = [result for result in results if result["status"] != "success"]
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())