#!/usr/bin/env python3
"""scripts/iree_assertion_stats.py

Collect hal.buffer_view.assert statistics for the IREE benchmark suite.

Two measurement modes
---------------------
1. **Static mode** (no rebuild required, default):
   Counts ``hal.buffer_view.assert`` ops in the pre-compiled
   ``benchmark/stablehlo-iree/executable-sources/`` files.  This gives the
   number of assertions that *survive* to the final program.

2. **Instrumented mode** (requires rebuilding IREE after patching
   ``Dialect/Stream/Conversion/HALToStream/Patterns.cpp``):
   Runs ``iree-compile --stats`` on each ``benchmark/iree/<cat>/*.mlir`` file
   and parses the LLVM statistics block for two counters added by the patch:
     - ``iree-hal-to-stream ... NumHALAssertionsGenerated``
     - ``iree-hal-to-stream ... NumHALAssertionsSkipped``
   The "surviving" count from static mode is also reported so we can compute
   the discharge count (generated − surviving).

Why two sources?
----------------
``hal.buffer_view.assert`` is generated during the HAL→Stream conversion pass
(``ConvertTensorImportOp::buildEncodingAssertions``).  There is currently
**no fold/canonicalize pattern** for this op in IREE's HAL dialect
(``HALOpFolders.cpp`` has no entry for it), so the generated count and the
surviving count should match.  The instrumented mode verifies this empirically
and would catch any future optimization that starts discharging assertions.

Output
------
  benchmark/results/iree_stats.csv
  columns: category, case_name, generated, surviving, discharged, notes

Usage
-----
  # Static mode (count from pre-compiled executable-sources):
  python scripts/iree_assertion_stats.py

  # Instrumented mode (requires rebuilt iree-compile with patch):
  python scripts/iree_assertion_stats.py --instrumented
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).parent.parent
IREE_COMPILE = (
    WORKSPACE_ROOT
    / "external/iree-git-v3.10.0/build-cuda-baseline/tools/iree-compile"
)
EXEC_SOURCES_DIR = WORKSPACE_ROOT / "benchmark/stablehlo-iree/executable-sources"
IREE_CASES_DIR = WORKSPACE_ROOT / "benchmark/iree"
DEFAULT_OUT = WORKSPACE_ROOT / "benchmark/results/iree_stats.csv"

_BV_ASSERT_RE = re.compile(r"\bhal\.buffer_view\.assert\b")

# Pattern for LLVM --stats output lines, e.g.:
#   3 iree-hal-to-stream - Number of hal.buffer_view.assert ops generated ...
_STATS_GENERATED_RE = re.compile(
    r"^\s*(\d+)\s+iree-hal-to-stream\s+-\s+Number of hal\.buffer_view\.assert ops generated",
    re.MULTILINE,
)
_STATS_SKIPPED_RE = re.compile(
    r"^\s*(\d+)\s+iree-hal-to-stream\s+-\s+Number of hal\.tensor\.import ops that skipped",
    re.MULTILINE,
)

# Cases IREE cannot compile due to dynamic shapes in matmul/conv2d
# (iree-compile errors on dynamically-shaped contraction dimensions)
_IREE_DYNAMIC_UNSUPPORTED_RE = re.compile(
    r"dynamic_.*x.*_.*x.*_|Nx.*_.*_|dynamic.*conv", re.IGNORECASE
)


def is_iree_unsupported(case_name: str) -> bool:
    """Heuristic: IREE cannot compile cases whose name contains a dynamic
    contraction dimension pattern typical of matmul/conv2d dynamic cases."""
    # Override: full list is determined at runtime by compilation failure.
    return False


def count_surviving(exec_src_file: Path) -> int | None:
    """Count hal.buffer_view.assert in a pre-compiled executable-sources file."""
    if not exec_src_file.exists():
        return None
    return len(_BV_ASSERT_RE.findall(exec_src_file.read_text(encoding="utf-8")))


def run_iree_instrumented(iree_compile: Path, input_file: Path) -> dict | None:
    """Run iree-compile --stats and parse the assertion generation counters.

    Returns dict with keys 'generated', 'skipped', or None on compile failure.
    """
    try:
        result = subprocess.run(
            [str(iree_compile),
             "--iree-input-type=none",
             "--iree-hal-target-backends=cuda",
             "--stats",
             str(input_file),
             "-o", "/dev/null"],
            capture_output=True, text=True,
            stdin=subprocess.DEVNULL,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {"generated": 0, "skipped": 0, "surviving": 0,
                "discharged": 0, "notes": "timeout"}
    combined = result.stdout + result.stderr

    if result.returncode != 0:
        # Check if this is a known "unsupported dynamic shape" failure
        if "dynamic" in input_file.name.lower() and (
            "matmul" in str(input_file).lower()
            or "conv" in str(input_file).lower()
        ):
            return {"generated": "N/A", "skipped": "N/A", "compile_error": "unsupported-dynamic-shape"}
        return None

    m_gen = _STATS_GENERATED_RE.search(combined)
    m_skip = _STATS_SKIPPED_RE.search(combined)
    generated = int(m_gen.group(1)) if m_gen else 0
    skipped = int(m_skip.group(1)) if m_skip else 0
    return {"generated": generated, "skipped": skipped, "compile_error": ""}


def collect_static(exec_sources_dir: Path, iree_cases_dir: Path) -> list[dict]:
    """Static mode: count from pre-compiled executable-sources."""
    rows: list[dict] = []
    for cat_dir in sorted(iree_cases_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        category = cat_dir.name
        exec_cat = exec_sources_dir / category
        for mlir_file in sorted(cat_dir.glob("*.mlir")):
            case_name = mlir_file.stem
            is_dynamic = int('dynamic' in case_name.lower() or
                          bool(re.search(r'(?:\d|x)([A-Z])(?:\d|x)', case_name)))
            # Match: benchmark/stablehlo-iree/executable-sources/<cat>/<case>.executable-sources.mlir
            exec_file = exec_cat / f"{case_name}.executable-sources.mlir"
            surviving = count_surviving(exec_file)
            if surviving is None:
                notes = "no-executable-sources-file"
                surviving_str = "N/A"
            else:
                notes = ""
                surviving_str = str(surviving)
            rows.append(dict(
                category=category,
                case_name=case_name,
                is_dynamic=is_dynamic,
                generated="unknown",    # requires instrumented mode
                surviving=surviving_str,
                discharged="unknown",   # requires instrumented mode
                notes=notes,
            ))
    return rows


def collect_instrumented(exec_sources_dir: Path, iree_cases_dir: Path,
                         iree_compile: Path,
                         verbose: bool = False) -> list[dict]:
    """Instrumented mode: run iree-compile --stats + cross-check with static count."""
    rows: list[dict] = []
    for cat_dir in sorted(iree_cases_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        category = cat_dir.name
        exec_cat = exec_sources_dir / category
        for mlir_file in sorted(cat_dir.glob("*.mlir")):
            case_name = mlir_file.stem
            is_dynamic = int('dynamic' in case_name.lower() or
                          bool(re.search(r'(?:\d|x)([A-Z])(?:\d|x)', case_name)))
            if verbose:
                print(f"  compiling {category}/{case_name} …", end="", flush=True)

            result = run_iree_instrumented(iree_compile, mlir_file)

            exec_file = exec_cat / f"{case_name}.executable-sources.mlir"
            surviving = count_surviving(exec_file)
            surviving_str = str(surviving) if surviving is not None else "N/A"

            if result is None:
                if verbose:
                    print(" FAIL")
                rows.append(dict(
                    category=category, case_name=case_name,
                    is_dynamic=is_dynamic,
                    generated="error", surviving=surviving_str,
                    discharged="error", notes="compile-error",
                ))
                continue

            if result.get("compile_error") == "unsupported-dynamic-shape":
                if verbose:
                    print(" N/A (unsupported dynamic shape)")
                rows.append(dict(
                    category=category, case_name=case_name,
                    is_dynamic=is_dynamic,
                    generated="N/A", surviving="N/A",
                    discharged="N/A", notes="unsupported-dynamic-shape",
                ))
                continue

            generated = result["generated"]
            if verbose:
                print(f" generated={generated} surviving={surviving_str}")

            if isinstance(generated, int) and isinstance(surviving, int):
                discharged = generated - surviving
                notes = result.get("compile_error", "")
            else:
                discharged = "unknown"
                notes = "mixed-types"

            rows.append(dict(
                category=category,
                case_name=case_name,
                is_dynamic=is_dynamic,
                generated=str(generated),
                surviving=surviving_str,
                discharged=str(discharged),
                notes=notes,
            ))
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instrumented", action="store_true",
                    help="Run iree-compile --stats (requires patched+rebuilt IREE)")
    ap.add_argument("--iree-compile", type=Path, default=IREE_COMPILE,
                    help="Path to iree-compile binary")
    ap.add_argument("--exec-sources-dir", type=Path, default=EXEC_SOURCES_DIR,
                    help="benchmark/stablehlo-iree/executable-sources/ directory")
    ap.add_argument("--iree-cases-dir", type=Path, default=IREE_CASES_DIR,
                    help="benchmark/iree/ directory (linalg+tensor cases)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)

    if not args.exec_sources_dir.is_dir():
        sys.exit(f"executable-sources dir not found: {args.exec_sources_dir}")
    if not args.iree_cases_dir.is_dir():
        sys.exit(f"IREE cases dir not found: {args.iree_cases_dir}")
    if args.instrumented and not args.iree_compile.exists():
        sys.exit(f"iree-compile not found: {args.iree_compile}")

    mode = "instrumented" if args.instrumented else "static"
    print(f"Mode: {mode}")

    if args.instrumented:
        rows = collect_instrumented(
            args.exec_sources_dir, args.iree_cases_dir,
            args.iree_compile, verbose=args.verbose,
        )
    else:
        rows = collect_static(args.exec_sources_dir, args.iree_cases_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "case_name", "is_dynamic", "generated", "surviving",
                  "discharged", "notes"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    numeric = [r for r in rows if r["surviving"] not in ("N/A", "error", "unknown")]
    total_surviving = sum(int(r["surviving"]) for r in numeric)
    print(f"Wrote {len(rows)} rows → {args.out}")
    print(f"  Total surviving runtime assertions: {total_surviving}")
    na_count = sum(1 for r in rows if r["surviving"] == "N/A")
    if na_count:
        print(f"  N/A cases (unsupported/missing): {na_count}")
    if args.instrumented:
        gen_rows = [r for r in rows if r["generated"] not in ("N/A", "error", "unknown")]
        total_gen = sum(int(r["generated"]) for r in gen_rows)
        total_dis = sum(int(r["discharged"]) for r in gen_rows
                        if r["discharged"] not in ("N/A", "error", "unknown"))
        print(f"  Total generated: {total_gen}")
        print(f"  Total discharged: {total_dis}")
        if total_gen > 0:
            print(f"  Resolution rate: {total_dis/total_gen*100:.1f}%")


if __name__ == "__main__":
    main()
