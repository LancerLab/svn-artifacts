#!/usr/bin/env python3
"""E2 — generation completeness (choreo-exact).

Plan §4.2: choreo is the only target whose safety assessments exist as an
*enumerable object*, so E2 is quantitative for choreo and qualitative
(expressibility only) for the SOTA toolchains. This script produces the
quantitative half: the complete obligation set for every kernel in the
15-category suite.

FEEDS  S3 (generation totals), S4 (per-operator ledger), S5 (discharge rate,
       split static vs dynamic), S6 (mechanism split).

--------------------------------------------------------------------------
THE LEDGER UNDERCOUNTS — AND WHY THIS SCRIPT COMBINES TWO SOURCES
--------------------------------------------------------------------------
`--dump-ledger` emits one JSON row per obligation that reaches the *assessor*.
But `--stats` also reports

    N  assess  - Direct static checks (bypassing assessor)

which are obligations discharged before the assessor ever sees them. They are
real obligations and they are part of the paper's headline count; they simply
have no ledger row. Measured on layer_normalization/3_attention_32xNx512x64_64_64:

    --stats  "Assessments evaluated"              79
    ledger   rows                                 72
    --stats  "Direct static checks"                7      72 + 7 = 79  ✓

    --stats  Assessments (element-access)         72   ledger elem-access  70  → 2 direct
    --stats  Assessments (shape-compatibility)     5   ledger shape-compat  0  → 5 direct
    --stats  Assessments (loop-bound)              2   ledger loop-bound    2  → 0 direct
                                                        total direct       = 7  ✓

So the per-class breakdown of the direct checks is *derivable* by differencing
the --stats per-usage counters against the ledger's per-usage counts. This
script does that and emits the derived checks as first-class obligation records
with `mechanism: direct` — which is exactly the enum value the record schema
asks for and which the ledger itself never emits. Reading only the ledger would
report 72 for layer_norm where the paper says 79, and would report the
`direct` mechanism as unpopulated.

Every kernel is reconciled: `Assessments evaluated == ledger_rows + direct`.
A kernel that fails to reconcile is flagged, never silently absorbed.

--------------------------------------------------------------------------
ENUM MAPPING (ledger → record schema)
--------------------------------------------------------------------------
The schema's enums and the ledger's do not line up one-to-one. Mapping used
here, proposed to the coordinator:

  usage            → class      elem-access→elem  shape-compat→shape
                                loop-bound→loop   hw-constraint→hw
  outcome          → outcome    static-true→proven  static-false→refuted
                                runtime+enabled=true →budgeted
                                runtime+enabled=false→runtime
  mechanism        → mechanism  canonical→canonical  interval→interval
                                (derived direct checks)→direct

`runtime` vs `budgeted`: a runtime obligation with `enabled: true` is emitted
into the generated code as an actual guard — it is discharged at runtime. One
with `enabled: false` was suppressed by the cost filter, so it is *accounted
for but not emitted*: budgeted. Measured invariant across the full 311-kernel
sweep: `enabled == (cost == "entry")` with 0 exceptions, i.e. only entry-cost
checks survive the filter.

--------------------------------------------------------------------------
STATIC vs DYNAMIC (needed for S5's split)
--------------------------------------------------------------------------
A kernel is dynamic-shape iff, after transitively resolving its `#define`d
integer macros, at least one extent in its `__co__` signature is still not a
compile-time integer literal. This matters because the suite's filenames lie in
both directions: `relu/1_bert_...` looks static and IS (`#define __STATIC_SHAPE__`
active, all extents macro-expanded to literals), while
`layer_normalization/3_attention_32xNx512x64_64_64` has no "dynamic" in its name
but carries a symbolic `N`. Name-based classification would put both in the
wrong bucket and corrupt S5.

The classification is cross-checked against the ledger: a kernel called static
that still emits runtime obligations is a contradiction and is flagged.

--------------------------------------------------------------------------
COST
--------------------------------------------------------------------------
The ledger fast path (`-gs --dump-ledger`, no nvcc) is 0.023 s per kernel
versus ~10 s for a full compile, and produces a byte-identical ledger. 311
kernels therefore sweep in seconds. Trap: adding `--no-codegen` makes choreo
emit NOTHING — do not use it.

E2 touches no GPU. exclusive=false.
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
import time

import toolchain                                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))          # benchmark2/choreo
B2 = os.path.dirname(HERE)                                 # benchmark2/
REPO = os.path.dirname(B2)                                 # svn-artifacts/

CHOREO = os.path.join(REPO, "croqtile", "build-release", "choreo")
SUITE = os.path.join(REPO, "benchmark", "choreo")
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(RAW, "e2_ledger.json")

TOOLCHAIN = "choreo"

# Pinned ledger invocation, manifest §3. Verbatim — reviewers re-run this.
LEDGER_FLAGS = ["-gs", "--stats", "-es",
                "--max-local-mem-capacity=2000000", "-t", "cute"]

T_LEDGER = 120

# ---------------------------------------------------------------- enums ----

USAGE_TO_CLASS = {
    "elem-access": "elem",
    "shape-compat": "shape",
    "loop-bound": "loop",
    "hw-constraint": "hw",
}

# --stats spells the classes out in prose; the ledger uses the short usage tag.
STATS_USAGE_LABEL = {
    "Assessments (element-access)": "elem-access",
    "Assessments (shape-compatibility)": "shape-compat",
    "Assessments (loop-bound)": "loop-bound",
    "Assessments (hw-constraint)": "hw-constraint",
    "Assessments (unclassified)": "unclassified",
}
STATS_RT_LABEL = {
    "Runtime assertions (element-access)": "elem-access",
    "Runtime assertions (shape-compatibility)": "shape-compat",
    "Runtime assertions (loop-bound)": "loop-bound",
    "Runtime assertions (hw-constraint)": "hw-constraint",
    "Runtime assertions (unclassified)": "unclassified",
}

RE_STAT = re.compile(r"^\s*(\d+)\s+assess\s+-\s+(.*?)\s*$")
RE_INFRA = re.compile(
    r"fatal error: .*No such file or directory"
    r"|cannot find -l"
    r"|CUDA_HOME|CUTE_HOME"
    r"|\[harness\] TIMEOUT"
    r"|No space left on device"
)


# ------------------------------------------------------------- utilities ----

def sha1_12(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


def settings_hash(category):
    """sha1 of benchmark2/settings/<category>.md — the cross-cutting field the
    statistics manifest calls non-negotiable."""
    p = os.path.join(B2, "settings", f"{category}.md")
    return sha1_12(p) if os.path.exists(p) else None


def toolchain_version():
    """The commit the choreo BINARY was built from — not the checkout HEAD.

    This used to be a bare `git rev-parse HEAD` of the croqtile checkout, per
    manifest §5. That is wrong whenever the checkout has moved since the binary
    was built, and on 2026-09-09 it produced records claiming a commit the
    running binary could not have come from (the binary was built 4 h 47 m
    before that commit existed). toolchain.py derives the version from the
    binary's mtime against the commit history and records the checkout HEAD
    separately, so the mismatch stays visible instead of being silently
    resolved in favour of the checkout. See that module for the full timeline.

    run_e2 owns this because E4 and E5 both call run_e2.toolchain_version();
    one definition means one answer across lanes.
    """
    return toolchain.version()


def run_ledger(src, ledger_path, out_sh, logpath):
    """Invoke the ledger fast path. Returns (rc, stdout+stderr text).

    cwd=REPO so relative paths in logs read sensibly; every path passed to
    choreo is absolute, because a relative one resolves against REPO and choreo
    reports "The input file ... does not exist" (rc=1).
    """
    import subprocess
    cmd = ["stdbuf", "-o0", "-e0", CHOREO, "-kt", src,
           f"--dump-ledger={ledger_path}", "-o", out_sh] + LEDGER_FLAGS
    with open(logpath, "wb") as lf:
        try:
            p = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT,
                               timeout=T_LEDGER)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            lf.write(b"\n[harness] TIMEOUT\n")
            rc = 124
    with open(logpath, "rb") as lf:
        return rc, lf.read().decode("utf-8", "replace")


def parse_stats(text):
    """The `--stats` assessment block → {label: count}."""
    out = {}
    for line in text.splitlines():
        m = RE_STAT.match(line)
        if m:
            out[m.group(2)] = int(m.group(1))
    return out


# ------------------------------------------------- static/dynamic shape ----

RE_DEFINE = re.compile(r"^\s*#\s*define\s+(\w+)\s*(.*)$")
RE_UNDEF = re.compile(r"^\s*#\s*undef\s+(\w+)")
RE_IFDEF = re.compile(r"^\s*#\s*ifdef\s+(\w+)")
RE_IFNDEF = re.compile(r"^\s*#\s*ifndef\s+(\w+)")
RE_IF = re.compile(r"^\s*#\s*if\s+(.+?)\s*$")
RE_ELSE = re.compile(r"^\s*#\s*else\b")
RE_ENDIF = re.compile(r"^\s*#\s*endif\b")
RE_CO_SIG = re.compile(r"__co__")
RE_BRACKETS = re.compile(r"\[([^\[\]]*)\]")
RE_INT_EXPR = re.compile(r"^[\d\s\+\-\*/\%\(\)]+$")


def preprocess(text):
    """Lightweight conditional-aware #define collector.

    Returns {NAME: replacement_text} for the macros that are actually in scope.

    WHY THIS IS NOT A ONE-LINER. The suite's static/dynamic switch is an
    #ifdef idiom, and reading #defines without honouring it inverts the answer.
    layer_normalization/3_attention_32xNx512x64_64_64:

        10: //#define __STATIC_SHAPE__     <-- commented out, so NOT defined
        11: #ifdef __STATIC_SHAPE__
        12: #define N0 J                   <-- must NOT be collected
        13: #endif

    A naive scan collects `N0 -> J`, resolves `J -> 32`, and declares the kernel
    static. It is dynamic: `N0` is a symbolic extent. 184 of the 311 suite files
    use this idiom; only 10 (all relu) have it active.

    The #else arm matters too. relu/1_bert_32x512x768_32x512x768 defines
    `__STATIC_SHAPE__` and takes the #ifdef arm, binding BATCH_SIZE to the
    literal 32; its #else arm binds BATCH_SIZE to the *symbol* `batch_size`.
    Picking the wrong arm flips the classification in the other direction.

    Directive forms present in the suite (exhaustive, by grep):
      #define 1803, #endif 486, #ifndef 279, #ifdef 206, #else 118,
      #undef 3, #if 1 (a literal `#if 0`). No #elif anywhere.
    """
    macros = {}
    # stack of (currently_taking, any_arm_taken)
    stack = []

    def active():
        return all(taking for taking, _ in stack)

    for line in text.splitlines():
        m = RE_IFDEF.match(line)
        if m:
            stack.append((m.group(1) in macros, False))
            continue
        m = RE_IFNDEF.match(line)
        if m:
            stack.append((m.group(1) not in macros, False))
            continue
        m = RE_IF.match(line)
        if m:
            expr = m.group(1).strip()
            taking = False
            if RE_INT_EXPR.match(expr):
                try:
                    taking = bool(eval(expr, {"__builtins__": {}}, {}))
                except Exception:
                    taking = False
            stack.append((taking, taking))
            continue
        if RE_ELSE.match(line):
            if stack:
                taking, seen = stack[-1]
                stack[-1] = (not taking and not seen, True)
            continue
        if RE_ENDIF.match(line):
            if stack:
                stack.pop()
            continue
        if not active():
            continue
        m = RE_UNDEF.match(line)
        if m:
            macros.pop(m.group(1), None)
            continue
        m = RE_DEFINE.match(line)
        if m:
            name, body = m.group(1), m.group(2).strip()
            # function-like macro: `#define F(a,b) ...` — the '(' immediately
            # follows the name with no space. Not an extent; skip it.
            if re.match(r"^\s*#\s*define\s+\w+\s*\(", line):
                continue
            macros[name] = body
    return macros


def resolve_value(name, macros, depth=0, seen=None):
    """Transitively resolve a macro to an int, or None if it stays symbolic."""
    if seen is None:
        seen = set()
    if name in seen or depth > 24 or name not in macros:
        return None
    seen = seen | {name}
    expr = macros[name]

    def sub(mo):
        v = resolve_value(mo.group(0), macros, depth + 1, seen)
        return str(v) if v is not None else mo.group(0)

    expr = re.sub(r"\b[A-Za-z_]\w*\b", sub, expr)
    if RE_INT_EXPR.match(expr):
        try:
            return int(eval(expr, {"__builtins__": {}}, {}))
        except Exception:
            return None
    return None


def signature_line(text):
    """The first `__co__` line carrying a `(` — the operator signature."""
    for line in text.splitlines():
        if RE_CO_SIG.search(line) and "(" in line:
            return line.strip()
    return None


# Scalar parameter types. A parameter of one of these types is a *runtime value*
# supplied by the caller, not a compile-time constant.
RE_SCALAR_TYPE = re.compile(
    r"^\s*(?:const\s+)?(?:unsigned\s+|signed\s+)?"
    r"(?:int|long|short|char|size_t|ssize_t|ptrdiff_t|"
    r"int8_t|int16_t|int32_t|int64_t|uint8_t|uint16_t|uint32_t|uint64_t)\b")


def split_params(sig):
    """Top-level comma split of the signature's parameter list.

    Depth-tracked so that `f32 [8, 128, 16, 16] i` stays one parameter instead
    of being split on the commas inside its extent brackets.
    """
    i = sig.find("(")
    if i < 0:
        return []
    depth = 0
    start = i + 1
    out = []
    for j in range(i, len(sig)):
        ch = sig[j]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                out.append(sig[start:j])
                return out
        elif ch == "," and depth == 1:
            out.append(sig[start:j])
            start = j + 1
    out.append(sig[start:])
    return out


def scalar_params(sig):
    """Names of parameters that are bare runtime scalars (no `[...]` extent).

    WHY THIS MATTERS. Reading only the bracket extents misclassifies conv2d.
    conv2d/10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D:

        __co__ auto CONV2D(f32 [8, 128, 16, 16] i, f32 [256, 128, 3, 3] w,
                           int stride, int padding, int dilation) {
          Ho = (H + 2*padding - dilation*(Kh-1) - 1)/stride + 1;
          f32 [N, Cout, Ho, Wo] o;

    Every *input* extent is an integer literal, so a bracket-only test calls the
    kernel static. It is not: the output extent `Ho` is derived from three
    runtime scalars, so no compile-time bound exists and the ledger correctly
    emits 11 runtime obligations. Eight conv2d kernels have this shape, and the
    bracket-only classifier flagged all eight as contradictions.
    """
    out = []
    for p in split_params(sig):
        p = p.strip()
        if not p or "[" in p:
            continue
        if RE_SCALAR_TYPE.match(p):
            out.append(p)
    return out


def is_dynamic(path):
    """(bool, signature, reasons) — see the module docstring.

    Dynamic iff EITHER
      (a) some extent in the resolved signature is not an integer literal, OR
      (b) the operator takes a bare scalar parameter (a runtime value that can
          feed a derived extent, as conv2d's stride/padding/dilation do).

    `reasons` lists the offending tokens so the report can show *why* a kernel
    was classed dynamic rather than just asserting it.
    """
    with open(path, "r", errors="replace") as f:
        text = f.read()
    sig = signature_line(text)
    if sig is None:
        return None, None, []
    macros = preprocess(text)

    def sub(mo):
        v = resolve_value(mo.group(0), macros)
        return str(v) if v is not None else mo.group(0)

    resolved = re.sub(r"\b[A-Za-z_]\w*\b", sub, sig)
    reasons = []
    for grp in RE_BRACKETS.findall(resolved):
        for tok in grp.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if not re.fullmatch(r"-?\d+", tok):
                reasons.append("extent:" + tok)
    for p in scalar_params(sig):
        reasons.append("scalar-param:" + re.sub(r"\s+", " ", p))
    return (len(reasons) > 0), sig, sorted(set(reasons))


# ------------------------------------------------------------- per kernel ----

def map_outcome(row):
    """ledger outcome (+ cost/enabled) → schema outcome enum.

    THE `enabled` DIRECTION, MEASURED NOT ASSUMED. Plan §5.2 splits the
    remainder into "runtime-materialized" and "budgeted-not-emitted", so the
    schema enum must follow which guards actually reach the generated code:

        enabled=true  → "runtime"   a real runtime_check(...) call is emitted
        enabled=false → "budgeted"  accounted for in the ledger, suppressed by
                                    the cost filter, no code emitted

    Verified on layer_normalization/3_attention_32xNx512x64_64_64, whose ledger
    has 5 runtime rows: 1 with cost=entry/enabled=true and 4 with
    cost=medium/enabled=false. Compiling with -rtc=entry emits exactly 6
    runtime_check sites — the 5 entry shape checks plus the single
    `zero is detected for the 1st dim` row that is enabled=true. The 4
    medium-cost rows emit nothing. -rtc=none emits 0. So enabled=true is the
    materialized set, and the earlier reading of this mapping was inverted.
    """
    o = row.get("outcome")
    if o == "static-true":
        return "proven"
    if o == "static-false":
        return "refuted"
    if o == "runtime":
        return "runtime" if row.get("enabled") else "budgeted"
    return "proven" if o is None else o


def process_kernel(category, case, workdir):
    kid = f"{category}/{case}"
    src = os.path.join(SUITE, category, f"{case}.co")
    d = os.path.join(workdir, kid.replace("/", "_"))
    os.makedirs(d, exist_ok=True)
    ledger_path = os.path.join(d, "ledger.json")
    out_sh = os.path.join(d, "k.sh")
    logpath = os.path.join(d, "ledger.log")

    rec = {
        "toolchain": TOOLCHAIN,
        "category": category,
        "kernel_id": kid,
        "path": os.path.relpath(src, REPO),
        "kernel_hash": sha1_12(src),
        "settings_hash": settings_hash(category),
    }

    dyn, sig, reasons = is_dynamic(src)
    rec["shape_class"] = "dynamic" if dyn else ("static" if dyn is False else "unknown")
    rec["signature"] = sig
    # Named `dynamic_reasons`, not `symbolic_extents`: the entries are tagged
    # `extent:<tok>` or `scalar-param:<decl>` because a bare scalar parameter
    # makes a kernel dynamic without any symbolic extent appearing in the
    # signature (conv2d's stride/padding/dilation). See is_dynamic().
    rec["dynamic_reasons"] = reasons

    if dyn is None:
        rec["compile"] = "fail"
        rec["note"] = "no __co__ signature found; not an operator kernel"
        rec["total_obligations"] = 0
        return rec

    rc, text = run_ledger(src, ledger_path, out_sh, logpath)
    rec["ledger_rc"] = rc

    if RE_INFRA.search(text):
        rec["compile"] = "fail"
        rec["infra_error"] = True
        rec["note"] = "infrastructure failure; verdict void"
        rec["total_obligations"] = 0
        return rec

    rows = []
    if os.path.exists(ledger_path) and os.path.getsize(ledger_path) > 0:
        try:
            rows = json.load(open(ledger_path)).get("obligations", [])
        except Exception as e:
            rec["compile"] = "fail"
            rec["note"] = f"ledger JSON unreadable: {e!r}"
            rec["total_obligations"] = 0
            return rec

    if rc != 0 and not rows:
        rec["compile"] = "fail"
        rec["note"] = f"choreo exited {rc} and emitted no ledger"
        rec["total_obligations"] = 0
        return rec

    rec["compile"] = "ok"
    stats = parse_stats(text)
    rec["stats"] = stats

    # ---- ledger rows → obligation records
    obligations = []
    per_usage = {}
    per_usage_rt = {}
    for i, row in enumerate(rows):
        usage = row.get("usage", "unclassified")
        cls = USAGE_TO_CLASS.get(usage, usage)
        oc = map_outcome(row)
        per_usage[usage] = per_usage.get(usage, 0) + 1
        if row.get("outcome") == "runtime":
            per_usage_rt[usage] = per_usage_rt.get(usage, 0) + 1
        obligations.append({
            "obligation_id": f"{kid}#{i:04d}",
            "class": cls,
            "outcome": oc,
            "mechanism": row.get("mechanism"),
            "dependence": row.get("dependence"),
            "cost": row.get("cost"),
            "enabled": row.get("enabled"),
            "loc": row.get("loc"),
            "message": row.get("message"),
            "source": "ledger",
        })

    # ---- direct checks: derived by differencing --stats against the ledger
    direct_total = stats.get("Direct static checks (bypassing assessor)", 0)
    derived = {}
    for label, usage in STATS_USAGE_LABEL.items():
        if label in stats:
            # --stats reports the per-class total INCLUDING the direct checks;
            # the ledger rows are the subset that reached the assessor.
            derived[usage] = max(0, stats[label] - per_usage.get(usage, 0))

    # prefer the per-class difference, but never exceed the reported total
    emitted_direct = 0
    for usage in ("elem-access", "shape-compat", "loop-bound", "hw-constraint",
                  "unclassified"):
        k = derived.get(usage, 0)
        if k <= 0:
            continue
        cls = USAGE_TO_CLASS.get(usage, usage)
        for j in range(k):
            if emitted_direct >= direct_total:
                break
            obligations.append({
                "obligation_id": f"{kid}#direct#{cls}#{j:02d}",
                "class": cls,
                "outcome": "proven",
                "mechanism": "direct",
                "dependence": None,
                "cost": None,
                "enabled": None,
                "loc": None,
                "message": "direct static check (bypassing assessor); class "
                           "derived by differencing --stats per-usage counters "
                           "against the ledger rows",
                "source": "stats-derived",
            })
            emitted_direct += 1

    rec["ledger_rows"] = len(rows)
    rec["direct_checks_reported"] = direct_total
    rec["direct_checks_emitted"] = emitted_direct
    rec["direct_by_class"] = {USAGE_TO_CLASS.get(u, u): v
                              for u, v in sorted(derived.items()) if v > 0}
    rec["obligations"] = obligations
    rec["total_obligations"] = len(obligations)

    # ---- reconciliation: the headline count must add up
    evaluated = stats.get("Assessments evaluated")
    rec["stats_evaluated"] = evaluated
    rec["reconciled"] = (evaluated is not None
                         and evaluated == len(rows) + direct_total
                         and emitted_direct == direct_total)
    if not rec["reconciled"]:
        rec["note"] = (f"RECONCILIATION FAILED: --stats evaluated={evaluated} "
                       f"but ledger_rows={len(rows)} + direct={direct_total} "
                       f"= {len(rows) + direct_total}; emitted_direct="
                       f"{emitted_direct}")

    # ---- cross-check the shape classification against the ledger
    n_runtime = sum(1 for o in obligations if o["source"] == "ledger"
                    and o["outcome"] in ("runtime", "budgeted"))
    rec["runtime_obligations"] = n_runtime
    if rec["shape_class"] == "static" and n_runtime > 0:
        rec["shape_contradiction"] = (
            f"classified static (all extents resolve to literals) yet the "
            f"ledger emits {n_runtime} runtime obligation(s)")

    # ---- per-class rollup, for S3/S4/S6 without re-walking the rows
    roll = {}
    for o in obligations:
        c = roll.setdefault(o["class"], {"proven": 0, "refuted": 0,
                                         "runtime": 0, "budgeted": 0,
                                         "canonical": 0, "interval": 0,
                                         "direct": 0, "total": 0})
        c[o["outcome"]] = c.get(o["outcome"], 0) + 1
        if o["mechanism"] in ("canonical", "interval", "direct"):
            c[o["mechanism"]] += 1
        c["total"] += 1
    rec["by_class"] = roll
    return rec


# ------------------------------------------------------------------- main ----

def all_kernels():
    """Every operator kernel in the suite: (category, case) pairs.

    EXCLUDES `relu/bench_relu`. It is not an operator case — it is a FileCheck
    compiler test (the only file in the suite carrying `// RUN:` lines, and it
    pins `-t factor` / `-t topscc`, not the manifest's `-t cute`). relu has 22
    `.co` files: 21 operator cases plus this test. Including it inflates the
    suite to 311 kernels and adds 192 obligations that belong to no operator;
    the prior benchmark's `choreo_stats.csv` has 310 rows for the same reason.
    Manifest §1 confirms 15 operator categories, and S3's grand total must count
    operator obligations only.
    """
    out = []
    for cat in sorted(os.listdir(SUITE)):
        d = os.path.join(SUITE, cat)
        if not os.path.isdir(d) or cat == "scripts":
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".co"):
                continue
            if name[:-3] == "bench_relu":
                continue
            out.append((cat, name[:-3]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only", default="", help="restrict to one category")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--size", default="small", choices=["small", "full"],
                    help="recorded for schema conformance; E2 is a compile-only "
                         "lane so the value does not change what is measured")
    ap.add_argument("--device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    ap.add_argument("--workdir", default=os.path.join(RAW, "e2_logs"))
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    # MUST be absolute: run_ledger() executes choreo with cwd=REPO.
    a.workdir = os.path.abspath(a.workdir)
    a.out = os.path.abspath(a.out)

    if not os.path.exists(CHOREO):
        print(f"ERROR: choreo binary not found at {CHOREO}", file=sys.stderr)
        print("       run `benchmark2/choreo/run.sh setup` first", file=sys.stderr)
        return 2

    kernels = all_kernels()
    if a.only:
        kernels = [(c, k) for c, k in kernels if c == a.only]
    if a.limit:
        kernels = kernels[:a.limit]

    os.makedirs(a.workdir, exist_ok=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    ver = toolchain_version()
    print(f"[choreo] E2: ledger sweep over {len(kernels)} kernels with "
          f"{a.jobs} workers (croqtile @ {ver or 'unknown'})")
    print(f"[choreo] pinned flags: {' '.join(LEDGER_FLAGS)}")

    results = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(process_kernel, c, k, a.workdir): (c, k)
                for c, k in kernels}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            c, k = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"toolchain": TOOLCHAIN, "category": c,
                     "kernel_id": f"{c}/{k}", "compile": "fail",
                     "total_obligations": 0,
                     "note": f"harness exception: {e!r}"}
            r["toolchain_version"] = ver
            r["gpu_device"] = str(a.device)
            r["exclusive"] = False          # compile-only lane, no GPU touched
            r["size"] = a.size
            results.append(r)
            if i % 25 == 0 or i == len(kernels):
                print(f"  [{i:>3}/{len(kernels)}] {r['kernel_id']:<56} "
                      f"{r['compile']:<4} obligations={r['total_obligations']}",
                      flush=True)

    results.sort(key=lambda x: x["kernel_id"])
    elapsed = round(time.time() - t0, 1)
    with open(a.out, "w") as f:
        json.dump({"toolchain": TOOLCHAIN, "toolchain_version": ver,
                   "produced_by": "e2", "elapsed_s": elapsed,
                   "gpu_device": str(a.device), "exclusive": False,
                   "pinned_flags": LEDGER_FLAGS,
                   "kernels": results}, f, indent=1)
    print(f"\nwrote {os.path.relpath(a.out, REPO)}  ({elapsed}s)")

    # ------------------------------------------------------------- report ----
    ok = [r for r in results if r.get("compile") == "ok"]
    bad = [r for r in results if r.get("compile") != "ok"]
    infra = [r for r in results if r.get("infra_error")]
    unreconciled = [r for r in ok if not r.get("reconciled")]
    contradicted = [r for r in ok if r.get("shape_contradiction")]

    total = sum(r.get("total_obligations", 0) for r in ok)
    ledger_total = sum(r.get("ledger_rows", 0) for r in ok)
    direct_total = sum(r.get("direct_checks_emitted", 0) for r in ok)

    print(f"\nkernels: {len(ok)} ok / {len(bad)} failed "
          f"({len(infra)} of those infrastructure)")
    print(f"obligations: {total} total = {ledger_total} ledger rows "
          f"+ {direct_total} direct checks (stats-derived)")

    # S3/S6 rollup
    roll = {}
    for r in ok:
        for cls, c in r.get("by_class", {}).items():
            t = roll.setdefault(cls, {"proven": 0, "refuted": 0, "runtime": 0,
                                      "budgeted": 0, "canonical": 0,
                                      "interval": 0, "direct": 0, "total": 0})
            for kk, vv in c.items():
                t[kk] = t.get(kk, 0) + vv

    print("\n=== S3 generation totals / S6 mechanism split ===")
    print(f"{'class':<8}{'total':>8}{'proven':>9}{'runtime':>9}{'budgeted':>10}"
          f"{'canon':>8}{'interval':>10}{'direct':>8}")
    for cls in ("elem", "shape", "loop", "hw", "unclassified"):
        if cls not in roll:
            continue
        c = roll[cls]
        print(f"{cls:<8}{c['total']:>8}{c['proven']:>9}{c['runtime']:>9}"
              f"{c['budgeted']:>10}{c['canonical']:>8}{c['interval']:>10}"
              f"{c['direct']:>8}")
    g = {k: sum(c[k] for c in roll.values()) for k in
         ("total", "proven", "refuted", "runtime", "budgeted",
          "canonical", "interval", "direct")}
    print(f"{'ALL':<8}{g['total']:>8}{g['proven']:>9}{g['runtime']:>9}"
          f"{g['budgeted']:>10}{g['canonical']:>8}{g['interval']:>10}"
          f"{g['direct']:>8}")

    # S5 discharge rate, split static vs dynamic
    print("\n=== S5 discharge rate (proven / total), split by shape class ===")
    for sc in ("static", "dynamic", "unknown"):
        sub = [r for r in ok if r.get("shape_class") == sc]
        if not sub:
            continue
        t = sum(r["total_obligations"] for r in sub)
        p = sum(c["proven"] for r in sub for c in r.get("by_class", {}).values())
        rate = (100.0 * p / t) if t else 0.0
        print(f"  {sc:<8} {len(sub):>3} kernels  {p:>6}/{t:<6} = {rate:5.1f}%")
    t = sum(r["total_obligations"] for r in ok)
    p = sum(c["proven"] for r in ok for c in r.get("by_class", {}).values())
    print(f"  {'ALL':<8} {len(ok):>3} kernels  {p:>6}/{t:<6} = "
          f"{(100.0 * p / t) if t else 0.0:5.1f}%")

    if unreconciled:
        print(f"\n*** {len(unreconciled)} KERNEL(S) FAILED TO RECONCILE "
              f"(--stats total != ledger rows + direct) ***")
        for r in unreconciled[:20]:
            print(f"    {r['kernel_id']}: {r.get('note')}")
    if contradicted:
        print(f"\n*** {len(contradicted)} SHAPE-CLASS CONTRADICTION(S) ***")
        for r in contradicted[:20]:
            print(f"    {r['kernel_id']}: {r['shape_contradiction']}")
    if bad:
        print(f"\n*** {len(bad)} KERNEL(S) DID NOT PRODUCE A LEDGER ***")
        for r in bad[:20]:
            print(f"    {r['kernel_id']}: {r.get('note', 'rc=' + str(r.get('ledger_rc')))}")
        if len(bad) > 20:
            print(f"    ... and {len(bad) - 20} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
