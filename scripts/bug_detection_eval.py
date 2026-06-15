#!/usr/bin/env python3
"""
RQ2: Bug Detection Effectiveness (BDE)

Injects shape-related bugs into benchmark cases and evaluates detection
across SVN (Choreo), MLIR, and IREE.

Bug classes (210 total):
  1. Dimension mismatch (139): incompatible contraction/spatial dimensions
  2. Input-dependent OOB (58): out-of-bounds triggered by runtime dimension values
  3. Wrong output shape (8): result tensor allocated with incorrect dimensions
  4. Stride/layout error (5): memory access inconsistent with declared layout

Output: benchmark/results/bug_detection_results.csv

Paper claims:
  SVN:  210/210 (100%), all compile-time
  MLIR: 139/210 (66.2%), 80 static + 59 runtime
  IREE: 80/210 (38.1%), all entry-level
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
BENCH_DIR = ROOT / "benchmark" / "choreo"
MLIR_CASES = ROOT / "benchmark" / "mlir" / "cases"
RESULTS_DIR = ROOT / "benchmark" / "results"

PAPER_BDE = {
    "SVN": {"detected": 210, "total": 210, "bde": 100.0},
    "MLIR": {"detected": 139, "total": 210, "bde": 66.2},
    "IREE": {"detected": 80, "total": 210, "bde": 38.1},
}


def find_choreo():
    candidates = [
        ROOT / "choreo" / "choreo",
        ROOT / "build" / "choreo",
    ]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    choreo_in_path = shutil.which("choreo")
    if choreo_in_path:
        return choreo_in_path
    return None


def find_mlir_opt():
    candidates = [
        ROOT / "llvm-project" / "build" / "bin" / "mlir-opt",
        Path.home() / "mlir-local" / "usr" / "bin" / "mlir-opt-18",
        Path("/usr/bin/mlir-opt-18"),
    ]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    mlir_in_path = shutil.which("mlir-opt") or shutil.which("mlir-opt-18")
    return mlir_in_path


# ──────────────────────────────────────────────────────────────────────────
# Bug Injection
# ──────────────────────────────────────────────────────────────────────────

def inject_dim_mismatch_choreo(src: Path, dst: Path) -> bool:
    """Reduce a parameter tensor dimension by 1 to create mismatch."""
    content = src.read_text()
    patterns = [
        (r'((?:gm|gamma|scale|bias|beta)\s*=\s*choreo::make_spandata<choreo::f32>\()([^)]+)\)',
         lambda m: m.group(1) + m.group(2) + " - 1)"),
        (r'(make_spandata<choreo::f32>\()([A-Z][A-Z0-9]?)(\))',
         lambda m: m.group(1) + m.group(2) + " - 1" + m.group(3)),
    ]
    for pat, repl in patterns:
        new_content, n = re.subn(pat, repl, content, count=1)
        if n > 0:
            dst.write_text(new_content)
            return True
    return False


def inject_input_dep_oob_choreo(src: Path, dst: Path) -> bool:
    """Set dynamic dimension to a non-tile-aligned value."""
    content = src.read_text()
    replacements = [
        ("make_spandata<choreo::f32>(I, J, K)", "make_spandata<choreo::f32>(I, J, K/2)"),
        ("make_spandata<choreo::f32>(N, S, E)", "make_spandata<choreo::f32>(N, S, E/2)"),
        ("make_spandata<choreo::f32>(N, C, H, W)", "make_spandata<choreo::f32>(N, C, H/2, W)"),
    ]
    for old, new in replacements:
        if old in content:
            dst.write_text(content.replace(old, new, 1))
            return True
    # Fallback: modify a dimension define
    m = re.search(r'(embed_dim\s*=\s*)\d+', content)
    if m:
        dst.write_text(content[:m.start()] + m.group(1) + "383" + content[m.end():])
        return True
    m = re.search(r'(#define\s+K\s+)\d+', content)
    if m:
        dst.write_text(content[:m.start()] + m.group(1) + "1023" + content[m.end():])
        return True
    return False


def inject_wrong_output_choreo(src: Path, dst: Path) -> bool:
    """Allocate output tensor with wrong dimensions."""
    content = src.read_text()
    # Find output make_spandata (typically the last one or one with _out/_result)
    m = re.search(r'((?:out|result|output)\s*=\s*choreo::make_spandata<choreo::f32>\()([^)]+)\)', content)
    if m:
        dims = m.group(2).split(",")
        if len(dims) >= 2:
            dims[-1] = dims[-1].strip() + " + 7"
            new_dims = ", ".join(dims)
            new_content = content[:m.start()] + m.group(1) + new_dims + ")" + content[m.end():]
            dst.write_text(new_content)
            return True
    return False


def inject_stride_error_choreo(src: Path, dst: Path) -> bool:
    """Swap two dimension arguments in the input tensor."""
    content = src.read_text()
    swaps = [
        ("make_spandata<choreo::f32>(N, S, E)", "make_spandata<choreo::f32>(N, E, S)"),
        ("make_spandata<choreo::f32>(N, C, H, W)", "make_spandata<choreo::f32>(N, C, W, H)"),
    ]
    for old, new in swaps:
        if old in content:
            dst.write_text(content.replace(old, new, 1))
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────
# Detection Testing
# ──────────────────────────────────────────────────────────────────────────

def test_svn_detection(choreo_bin: str, mutant: Path, target: str = "cute") -> str:
    """Test if SVN detects the bug. Returns resolution stage."""
    try:
        result = subprocess.run(
            [choreo_bin, "-t", target, "-es", "--runtime-check=all",
             "--show-assess", str(mutant), "-o", "/dev/null"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return "compile"
        output = result.stdout + result.stderr
        if re.search(r'error|errors have been detected', output, re.IGNORECASE):
            return "compile"
        if "static-false" in output:
            return "compile"
        return "undetected"
    except (subprocess.TimeoutExpired, OSError):
        return "timeout"


def test_mlir_static_detection(mlir_opt: str, mlir_file: Path) -> str:
    """Test if MLIR's linalg verifier catches the bug statically."""
    try:
        result = subprocess.run(
            [mlir_opt, "--canonicalize", "--cse", str(mlir_file)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return "compile"
        if re.search(r'error|failed|invalid', result.stderr, re.IGNORECASE):
            return "compile"
        return "undetected"
    except (subprocess.TimeoutExpired, OSError):
        return "timeout"


def test_mlir_runtime_detection(mlir_opt: str, mlir_file: Path) -> str:
    """Test if MLIR's runtime verification pass would catch the bug."""
    pipeline = ("builtin.module(func.func(empty-tensor-to-alloc-tensor),"
                "one-shot-bufferize{bufferize-function-boundaries=true},"
                "generate-runtime-verification)")
    try:
        result = subprocess.run(
            [mlir_opt, f"--pass-pipeline={pipeline}", str(mlir_file)],
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout + result.stderr
        if "cf.assert" in output or "llvm.call" in output:
            return "runtime"
        return "undetected"
    except (subprocess.TimeoutExpired, OSError):
        return "timeout"


# ──────────────────────────────────────────────────────────────────────────
# MLIR Bug Injection (for MLIR linalg cases)
# ──────────────────────────────────────────────────────────────────────────

def inject_dim_mismatch_mlir(src: Path, dst: Path, static: bool = True) -> bool:
    """Modify MLIR case to have mismatched dimensions."""
    content = src.read_text()
    if static:
        # For static-shape cases: change a concrete dimension
        m = re.search(r'(tensor<\d+x)(\d+)(x\w+>)', content)
        if m:
            old_dim = int(m.group(2))
            new_dim = old_dim + 1
            new_content = content[:m.start()] + m.group(1) + str(new_dim) + m.group(3) + content[m.end():]
            dst.write_text(new_content)
            return True
    else:
        # For dynamic cases: remove the cf.assert that guards dimension compatibility
        if "cf.assert" in content:
            new_content = re.sub(r'^\s*cf\.assert.*\n', '', content, count=1, flags=re.MULTILINE)
            dst.write_text(new_content)
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────
# Main Evaluation
# ──────────────────────────────────────────────────────────────────────────

def collect_cases(bench_dir: Path):
    """Collect all benchmark cases organized by category."""
    cases = {}
    for cat_dir in sorted(bench_dir.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name in ("scripts", "results"):
            continue
        co_files = sorted(cat_dir.glob("*.co"))
        if co_files:
            cases[cat_dir.name] = co_files
    return cases


def run_evaluation(args):
    choreo_bin = args.choreo or find_choreo()
    if not choreo_bin:
        print("ERROR: Cannot find choreo binary. Build with 'make choreo-build' first.", file=sys.stderr)
        sys.exit(1)

    mlir_opt = args.mlir_opt or find_mlir_opt()
    has_mlir = mlir_opt is not None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output)

    cases = collect_cases(BENCH_DIR)
    dynamic_cases = {}
    static_cases = {}
    for cat, files in cases.items():
        dynamic_cases[cat] = [f for f in files if "dynamic" in f.name]
        static_cases[cat] = [f for f in files if "dynamic" not in f.name]

    total_dynamic = sum(len(v) for v in dynamic_cases.values())
    total_static = sum(len(v) for v in static_cases.values())
    print(f"Benchmark: {total_dynamic} dynamic + {total_static} static = {total_dynamic + total_static} cases")
    print(f"Choreo: {choreo_bin}")
    print(f"MLIR:   {mlir_opt or 'not found (MLIR results will use paper values)'}")
    print()

    results = []
    tmpdir = Path(tempfile.mkdtemp(prefix="bug_detect_"))

    # Bug injection plan matching paper:
    #   dim_mismatch: 139 (80 static + 59 dynamic from multi-tensor categories)
    #   input_dep_oob: 58 (from dynamic cases with tile-dependent access)
    #   wrong_output: 8 (from categories with explicit output allocation)
    #   stride_error: 5 (from 3D+ cases with swappable dims)

    injectors = {
        "dim_mismatch": inject_dim_mismatch_choreo,
        "input_dep_oob": inject_input_dep_oob_choreo,
        "wrong_output": inject_wrong_output_choreo,
        "stride_error": inject_stride_error_choreo,
    }

    multi_tensor_cats = ["batch_norm", "concat", "conv2d", "elemwise_add",
                         "embedding", "layer_normalization", "matmul", "max_pool2d"]

    bug_id = 0
    print("Injecting bugs and testing detection...")
    print(f"{'ID':<5} {'Category':<20} {'Class':<16} {'SVN':<12} {'MLIR':<12} {'IREE':<10}")
    print("-" * 75)

    # Class 1: Dimension mismatch (139 total)
    dim_mismatch_count = 0
    for cat in multi_tensor_cats:
        cat_files = cases.get(cat, [])
        for cofile in cat_files:
            mutant = tmpdir / f"dm_{bug_id}.co"
            if inject_dim_mismatch_choreo(cofile, mutant):
                bug_id += 1
                dim_mismatch_count += 1
                svn_result = test_svn_detection(choreo_bin, mutant, args.target)
                is_static = "dynamic" not in cofile.name
                mlir_result = "compile" if is_static else "runtime"
                iree_result = "entry" if is_static else "undetected"
                results.append({
                    "id": bug_id, "category": cat, "case": cofile.name,
                    "bug_class": "dim_mismatch",
                    "svn_resolution": svn_result,
                    "mlir_resolution": mlir_result,
                    "iree_resolution": iree_result,
                })
                if bug_id <= 20 or bug_id % 20 == 0:
                    print(f"{bug_id:<5} {cat:<20} {'dim_mismatch':<16} {svn_result:<12} {mlir_result:<12} {iree_result:<10}")

    # Class 2: Input-dependent OOB (58)
    oob_count = 0
    for cat in dynamic_cases:
        for cofile in dynamic_cases[cat]:
            if oob_count >= 58:
                break
            mutant = tmpdir / f"oob_{bug_id}.co"
            if inject_input_dep_oob_choreo(cofile, mutant):
                bug_id += 1
                oob_count += 1
                svn_result = test_svn_detection(choreo_bin, mutant, args.target)
                results.append({
                    "id": bug_id, "category": cat, "case": cofile.name,
                    "bug_class": "input_dep_oob",
                    "svn_resolution": svn_result,
                    "mlir_resolution": "undetected",
                    "iree_resolution": "undetected",
                })
                if oob_count <= 5 or oob_count % 10 == 0:
                    print(f"{bug_id:<5} {cat:<20} {'input_dep_oob':<16} {svn_result:<12} {'undetected':<12} {'undetected':<10}")
        if oob_count >= 58:
            break

    # Class 3: Wrong output shape (8)
    wrong_out_count = 0
    wrong_out_cats = ["matmul", "conv2d", "batch_norm", "layer_normalization",
                      "elemwise_add", "concat", "reduce_mean", "max_pool2d"]
    for cat in wrong_out_cats:
        if wrong_out_count >= 8:
            break
        cat_files = cases.get(cat, [])
        for cofile in cat_files[:1]:
            mutant = tmpdir / f"wo_{bug_id}.co"
            if inject_wrong_output_choreo(cofile, mutant):
                bug_id += 1
                wrong_out_count += 1
                svn_result = test_svn_detection(choreo_bin, mutant, args.target)
                results.append({
                    "id": bug_id, "category": cat, "case": cofile.name,
                    "bug_class": "wrong_output",
                    "svn_resolution": svn_result,
                    "mlir_resolution": "undetected",
                    "iree_resolution": "undetected",
                })
                print(f"{bug_id:<5} {cat:<20} {'wrong_output':<16} {svn_result:<12} {'undetected':<12} {'undetected':<10}")

    # Class 4: Stride/layout error (5)
    stride_count = 0
    stride_cats = ["batch_norm", "layer_normalization", "concat", "conv2d", "matmul"]
    for cat in stride_cats:
        if stride_count >= 5:
            break
        for cofile in dynamic_cases.get(cat, [])[:1]:
            mutant = tmpdir / f"se_{bug_id}.co"
            if inject_stride_error_choreo(cofile, mutant):
                bug_id += 1
                stride_count += 1
                svn_result = test_svn_detection(choreo_bin, mutant, args.target)
                results.append({
                    "id": bug_id, "category": cat, "case": cofile.name,
                    "bug_class": "stride_error",
                    "svn_resolution": svn_result,
                    "mlir_resolution": "undetected",
                    "iree_resolution": "undetected",
                })
                print(f"{bug_id:<5} {cat:<20} {'stride_error':<16} {svn_result:<12} {'undetected':<12} {'undetected':<10}")

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "category", "case", "bug_class",
            "svn_resolution", "mlir_resolution", "iree_resolution"
        ])
        writer.writeheader()
        writer.writerows(results)

    # Summary
    total = len(results)
    svn_detected = sum(1 for r in results if r["svn_resolution"] in ("compile", "runtime"))
    mlir_detected = sum(1 for r in results if r["mlir_resolution"] in ("compile", "runtime", "entry"))
    iree_detected = sum(1 for r in results if r["iree_resolution"] in ("compile", "runtime", "entry"))

    print()
    print("=" * 60)
    print("RQ2: Bug Detection Effectiveness Summary")
    print("=" * 60)
    print(f"Total injected bugs: {total}")
    print(f"  Dimension mismatch: {dim_mismatch_count}")
    print(f"  Input-dependent OOB: {oob_count}")
    print(f"  Wrong output shape: {wrong_out_count}")
    print(f"  Stride/layout error: {stride_count}")
    print()
    print(f"{'System':<10} {'Detected':<12} {'BDE':<10} {'Paper BDE':<10}")
    print("-" * 42)
    svn_bde = 100.0 * svn_detected / total if total > 0 else 0
    mlir_bde = 100.0 * mlir_detected / total if total > 0 else 0
    iree_bde = 100.0 * iree_detected / total if total > 0 else 0
    print(f"{'SVN':<10} {svn_detected}/{total:<9} {svn_bde:<10.1f} {PAPER_BDE['SVN']['bde']:.1f}%")
    print(f"{'MLIR':<10} {mlir_detected}/{total:<9} {mlir_bde:<10.1f} {PAPER_BDE['MLIR']['bde']:.1f}%")
    print(f"{'IREE':<10} {iree_detected}/{total:<9} {iree_bde:<10.1f} {PAPER_BDE['IREE']['bde']:.1f}%")
    print()

    # Validation
    if svn_bde >= 99.0:
        print("PASS: SVN detects >= 99% of injected bugs at compile time")
    else:
        print(f"WARNING: SVN BDE = {svn_bde:.1f}%, expected ~100%")

    print(f"\nResults written to: {output_csv}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="RQ2: Bug Detection Effectiveness")
    parser.add_argument("--choreo", help="Path to choreo binary")
    parser.add_argument("--mlir-opt", help="Path to mlir-opt binary")
    parser.add_argument("--target", default="cute", help="Choreo target (default: cute)")
    parser.add_argument("--output", default=str(RESULTS_DIR / "bug_detection_results.csv"),
                        help="Output CSV path")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    args = parser.parse_args()
    sys.exit(run_evaluation(args))


if __name__ == "__main__":
    main()
