#!/usr/bin/env python3
"""scripts/triton_assertion_stats.py

Count explicit assertions in the Triton benchmark cases
(benchmark/triton/<category>/<case>.py).

Triton assertion model
----------------------
Triton is a GPU kernel DSL with memory-level semantics only.  It has no
built-in type system for tensor shape constraints.  All safety checks must be
written as explicit programmer assertions.  The benchmark cases mark each
assertion with a ``# EXPLICIT_ASSERTION:`` comment immediately before the
``require(...)`` call, making them straightforward to count.

Triton's compiler does insert one *compiler-generated* assertion: an integer
overflow check via ``tl.device_assert`` when ``sanitize_overflow=True`` fires
for <64-bit integer arithmetic.  For our float32-dominated benchmark workloads
this path is NOT triggered (the guard is
``if lhs_sca_ty.int_bitwidth >= 64 or not self.builder.options.sanitize_overflow``),
so compiler_internal is reliably 0.  Nonetheless this script also counts
``tl.device_assert`` calls in the source as a cross-check (``tl_device_assert``
column).

Discharge model
---------------
Triton performs NO compile-time symbolic shape reasoning; every explicit
assertion becomes a runtime check.  Therefore:
  discharged = 0
  runtime    = explicit_assertions

Metric columns
--------------
  category, case_name,
  explicit_assertions  : # EXPLICIT_ASSERTION: markers (programmer-written)
  tl_device_assert     : tl.device_assert() calls in source (compiler-gen proxy)
  generated_total      : explicit_assertions + tl_device_assert
  discharged           : 0 (no compile-time discharge)
  runtime              : same as generated_total
  resolution_rate      : 0.0000

Outputs
-------
  benchmark/results/triton_stats.csv

Usage
-----
  python scripts/triton_assertion_stats.py [--cases-dir DIR] [--out CSV]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

WORKSPACE_ROOT    = Path(__file__).parent.parent
DEFAULT_CASES_DIR = WORKSPACE_ROOT / "benchmark/triton"
DEFAULT_OUT       = WORKSPACE_ROOT / "benchmark/results/triton_stats.csv"

_EXPLICIT_RE     = re.compile(r"#\s*EXPLICIT_ASSERTION:")
_TL_DEV_ASSERT_RE = re.compile(r"\btl\.device_assert\s*\(")


def count_file(py_file: Path) -> dict:
    text = py_file.read_text(encoding="utf-8")
    explicit     = len(_EXPLICIT_RE.findall(text))
    tl_dev       = len(_TL_DEV_ASSERT_RE.findall(text))
    generated    = explicit + tl_dev
    return {
        "explicit_assertions": explicit,
        "tl_device_assert":    tl_dev,
        "generated_total":     generated,
        "discharged":          0,
        "runtime":             generated,
        "resolution_rate":     "0.0000",
    }


def collect(cases_dir: Path) -> list[dict]:
    rows: list[dict] = []
    # Walk cat_dir/*, skip special dirs
    _SKIP = {"cases", "tests", "results"}
    for cat_dir in sorted(cases_dir.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name in _SKIP:
            continue
        category = cat_dir.name
        for py_file in sorted(cat_dir.glob("*.py")):
            case_name = py_file.stem
            is_dynamic = int('dynamic' in case_name.lower() or
                          bool(re.search(r'(?:\d|x)([A-Z])(?:\d|x)', case_name)))
            stats = count_file(py_file)
            rows.append(dict(category=category, case_name=case_name,
                             is_dynamic=is_dynamic, **stats))
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    if not args.cases_dir.is_dir():
        sys.exit(f"Triton cases dir not found: {args.cases_dir}")

    print(f"Scanning {args.cases_dir} …")
    rows = collect(args.cases_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "case_name", "is_dynamic",
                  "explicit_assertions", "tl_device_assert",
                  "generated_total", "discharged", "runtime", "resolution_rate"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_exp = sum(r["explicit_assertions"] for r in rows)
    total_gen = sum(r["generated_total"]     for r in rows)
    total_tld = sum(r["tl_device_assert"]    for r in rows)
    print(f"Wrote {len(rows)} rows → {args.out}")
    print(f"  Total explicit assertions : {total_exp}")
    print(f"  Total tl.device_assert    : {total_tld}")
    print(f"  Total generated           : {total_gen}")
    print(f"  Total discharged          : 0  (0.0% — no compile-time discharge)")
    print(f"  Total runtime             : {total_gen}")


if __name__ == "__main__":
    main()
