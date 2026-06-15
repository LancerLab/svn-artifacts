#!/usr/bin/env python3
"""scripts/mlir_assertion_stats.py

Measure cf.assert generation and compile-time discharge for the MLIR/tensor+scf
benchmark suite by running mlir-opt with two measurement layers:

Layer 1 – User assertions (existing behaviour)
-----------------------------------------------
Count ``cf.assert`` ops present in the hand-written benchmark source.  These
are the constraints that the programmer had to write explicitly (equivalent to
Triton's explicit assertions).

Layer 2 – Internal RuntimeOpVerification assertions (new)
----------------------------------------------------------
Run ``generate-runtime-verification`` first (adds compiler-generated bound/rank
checks for tensor.extract/insert/empty/reshape etc.), then count the total
(user + internal = "generated").  A subsequent ``canonicalize,cse`` step folds
assertions whose condition is statically-known-true, giving "surviving" =
runtime count.  discharged = generated − runtime.

Metric summary
--------------
  generated_user     : cf.assert in hand-written source
  generated_internal : additional cf.assert from generate-runtime-verification
  generated_total    : generated_user + generated_internal
  discharged         : generated_total − runtime  (static-true fold)
  runtime            : surviving after canonicalize,cse
  resolution_rate    : discharged / generated_total

Outputs
-------
  benchmark/results/mlir_stats.csv

Usage
-----
  python scripts/mlir_assertion_stats.py [--cases-dir DIR] [--out CSV]
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).parent.parent
MLIR_OPT = WORKSPACE_ROOT / "build/llvm-release/bin/mlir-opt"
DEFAULT_CASES_DIR = WORKSPACE_ROOT / "benchmark/mlir/cases"
DEFAULT_OUT = WORKSPACE_ROOT / "benchmark/results/mlir_stats.csv"

# Pass pipelines
_GEN_RV_PIPELINE      = "builtin.module(func.func(generate-runtime-verification))"
_DISCHARGE_PIPELINE   = "builtin.module(func.func(generate-runtime-verification,canonicalize,cse))"
_USER_DISCHARGE_PIPE  = "builtin.module(func.func(canonicalize,cse))"

_CF_ASSERT_RE = re.compile(r"\bcf\.assert\b")


def count_cf_assert(text: str) -> int:
    return len(_CF_ASSERT_RE.findall(text))


_MLIR_OPT_TIMEOUT_S = 120

def run_mlir_opt(mlir_opt: Path, input_path: Path, pipeline: str) -> str | None:
    """Run mlir-opt with the given pipeline; return stdout or None on error."""
    try:
        result = subprocess.run(
            [str(mlir_opt), f"--pass-pipeline={pipeline}", str(input_path), "-o", "-"],
            capture_output=True, text=True,
            stdin=subprocess.DEVNULL,
            timeout=_MLIR_OPT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def collect(cases_dir: Path, mlir_opt: Path) -> list[dict]:
    rows: list[dict] = []
    for cat_dir in sorted(cases_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        category = cat_dir.name
        for mlir_file in sorted(cat_dir.glob("*.mlir")):
            case_name = mlir_file.stem
            text_src = mlir_file.read_text(encoding="utf-8")
            is_dynamic = int('dynamic' in case_name.lower() or
                          bool(re.search(r'(?:\d|x)([A-Z])(?:\d|x)', case_name)))

            # Layer 1: user-written assertions
            generated_user = count_cf_assert(text_src)

            # Layer 2a: total after RuntimeOpVerification (user + internal)
            text_after_rv = run_mlir_opt(mlir_opt, mlir_file, _GEN_RV_PIPELINE)
            if text_after_rv is None:
                print(f"  WARN: generate-runtime-verification failed on "
                      f"{mlir_file.relative_to(WORKSPACE_ROOT)}", file=sys.stderr)
                rows.append(dict(category=category, case_name=case_name,
                                 is_dynamic=is_dynamic,
                                 generated_user=generated_user,
                                 generated_internal="error",
                                 generated_total="error",
                                 discharged="error", runtime="error",
                                 resolution_rate="error"))
                continue
            generated_total = count_cf_assert(text_after_rv)
            generated_internal = generated_total - generated_user

            # Layer 2b: surviving after discharge (canonicalize + CSE)
            text_after_discharge = run_mlir_opt(mlir_opt, mlir_file, _DISCHARGE_PIPELINE)
            if text_after_discharge is None:
                print(f"  WARN: discharge pipeline failed on "
                      f"{mlir_file.relative_to(WORKSPACE_ROOT)}", file=sys.stderr)
                rows.append(dict(category=category, case_name=case_name,
                                 is_dynamic=is_dynamic,
                                 generated_user=generated_user,
                                 generated_internal=generated_internal,
                                 generated_total=generated_total,
                                 discharged="error", runtime="error",
                                 resolution_rate="error"))
                continue

            runtime = count_cf_assert(text_after_discharge)
            discharged = generated_total - runtime
            resolution_rate = (discharged / generated_total) if generated_total > 0 else 1.0
            rows.append(dict(
                category=category,
                case_name=case_name,
                is_dynamic=is_dynamic,
                generated_user=generated_user,
                generated_internal=generated_internal,
                generated_total=generated_total,
                discharged=discharged,
                runtime=runtime,
                resolution_rate=f"{resolution_rate:.4f}",
            ))
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES_DIR,
                    help="Root of MLIR benchmark cases (default: benchmark/mlir/cases)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="Output CSV path (default: benchmark/results/mlir_stats.csv)")
    ap.add_argument("--mlir-opt", type=Path, default=MLIR_OPT,
                    help="Path to mlir-opt binary")
    args = ap.parse_args(argv)

    if not args.mlir_opt.exists():
        sys.exit(f"mlir-opt not found at {args.mlir_opt}. Run 'make mlir-build' first.")
    if not args.cases_dir.is_dir():
        sys.exit(f"Cases dir not found: {args.cases_dir}")

    print(f"Scanning {args.cases_dir} …")
    rows = collect(args.cases_dir, args.mlir_opt)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "case_name", "is_dynamic", "generated_user", "generated_internal",
                  "generated_total", "discharged", "runtime", "resolution_rate"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    ok = [r for r in rows if r["discharged"] != "error"]
    total_gen  = sum(int(r["generated_total"]) for r in ok)
    total_user = sum(int(r["generated_user"])  for r in ok)
    total_int  = sum(int(r["generated_internal"]) for r in ok)
    total_dis  = sum(int(r["discharged"]) for r in ok)
    total_rt   = sum(int(r["runtime"]) for r in ok)
    overall    = total_dis / total_gen if total_gen else 1.0
    print(f"Wrote {len(rows)} rows → {args.out}")
    print(f"  Total generated (user)    : {total_user}")
    print(f"  Total generated (internal): {total_int}")
    print(f"  Total generated (total)   : {total_gen}")
    print(f"  Total discharged          : {total_dis}  ({overall*100:.1f}% resolution rate)")
    print(f"  Total runtime             : {total_rt}")


if __name__ == "__main__":
    main()
