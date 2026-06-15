#!/usr/bin/env python3
"""Run all Triton benchmark cases with a single command.

Modes:
  --mode compile-only   JIT smoke test per case (fast, default)
  --mode chunk-check    correctness check per case (slower, requires GPU)

Exit 0 if all executed cases succeed (n/a is not failure).
Exit 1 if any case returns an unexpected error.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRITON_ROOT = ROOT / "benchmark" / "triton"
CASES_DIR = TRITON_ROOT / "cases"
SKIP_DIRS = {"cases", "tests", "results", "__pycache__", "scripts"}

if str(CASES_DIR) not in sys.path:
    sys.path.insert(0, str(CASES_DIR))


def discover_cases(category_filter: str | None = None) -> list[tuple[str, Path]]:
    cases = []
    for cat_dir in sorted(TRITON_ROOT.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name in SKIP_DIRS:
            continue
        if category_filter and cat_dir.name != category_filter:
            continue
        for case_file in sorted(cat_dir.glob("*.py")):
            if case_file.stem[0].isdigit():
                cases.append((cat_dir.name, case_file))
    return cases


def count_assertions(case_file: Path) -> int:
    return case_file.read_text().count("# EXPLICIT_ASSERTION:")


def load_and_run(case_file: Path, *, chunk_check: bool, compile_only: bool) -> dict:
    spec = importlib.util.spec_from_file_location(f"_tcase_{case_file.stem}", case_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    env = dict(mod.SYMBOL_DEFAULTS)
    return mod.run_case(env, chunk_check=chunk_check, compile_only=compile_only)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("compile-only", "chunk-check"),
                        default="compile-only",
                        help="compile-only: JIT smoke (default); chunk-check: correctness")
    parser.add_argument("--category", default=None,
                        help="Run only this category (e.g. batch_norm)")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Emit one JSON object per line instead of human-readable output")
    args = parser.parse_args()

    cases = discover_cases(args.category)
    chunk_check = args.mode == "chunk-check"
    compile_only = args.mode == "compile-only"

    passed = failed = skipped = 0
    total_assertions = 0
    by_cat: dict[str, dict] = {}
    results: list[dict] = []

    for category, case_file in cases:
        assertions = count_assertions(case_file)
        total_assertions += assertions
        try:
            result = load_and_run(case_file, chunk_check=chunk_check, compile_only=compile_only)
        except Exception as exc:
            result = {
                "status": "error",
                "mode": args.mode,
                "input_shape": "?",
                "output_shape": "?",
                "time_ms": 0.0,
                "notes": repr(exc),
            }
            traceback.print_exc(file=sys.stderr)

        result["category"] = category
        result["case_name"] = case_file.stem
        result["explicit_assertions"] = assertions

        st = result["status"]
        if st == "success":
            passed += 1
        elif st == "n/a":
            skipped += 1
        else:
            failed += 1

        cat_stats = by_cat.setdefault(category, {"ok": 0, "na": 0, "err": 0, "assertions": 0})
        cat_stats[{"success": "ok", "n/a": "na"}.get(st, "err")] += 1
        cat_stats["assertions"] += assertions

        if args.as_json:
            print(json.dumps(result, separators=(",", ":")))
        else:
            tag = "OK " if st == "success" else ("N/A" if st == "n/a" else "ERR")
            notes = result.get("notes", "")
            print(f"[{tag}] {category}/{case_file.stem}  {notes}")

        results.append(result)

    if not args.as_json:
        print(f"\n{'=' * 62}")
        print(f"  Cases : {len(cases)}   Passed: {passed}   N/A: {skipped}   Failed: {failed}")
        print(f"  Total explicit assertions: {total_assertions}")
        print(f"  Mode  : {args.mode}")
        print(f"{'=' * 62}")
        for cat, c in sorted(by_cat.items()):
            print(f"  {cat:25s}  ok={c['ok']:3d}  n/a={c['na']:3d}  "
                  f"err={c['err']:3d}  assertions={c['assertions']}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
