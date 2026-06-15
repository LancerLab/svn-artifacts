#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys

from triton_case_common import describe_case, run_case


def bound_case_main(case_path: str, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a generated Triton-host benchmark case")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--describe-case", action="store_true")
    args = parser.parse_args(argv)

    if args.describe_case:
        print(json.dumps(describe_case(case_path), separators=(",", ":")))
        return 0

    print(json.dumps(run_case(case_path, device=args.device), separators=(",", ":")))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a generated Triton-host benchmark case")
    parser.add_argument("--case", required=True, help="Path to the Choreo benchmark case")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--describe-case", action="store_true")
    args = parser.parse_args()
    forwarded = ["--device", args.device]
    if args.describe_case:
        forwarded.append("--describe-case")
    return bound_case_main(args.case, forwarded)


if __name__ == "__main__":
    sys.exit(main())