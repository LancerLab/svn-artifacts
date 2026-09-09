#!/usr/bin/env python3
"""Shared harness for the benchmark2 MLIR lanes (`mlir-linalg`, `mlir-low`).

Both lanes compose their own kernels from `benchmark2/settings/`, mutate them per
`benchmark2/specs/mutation-specs.md`, and classify every mutant into the
`{compile, runtime, never, n/a}` taxonomy with the §7 ground-truth oracle.

This module owns everything that must be *identical* across the two lanes so the
numbers are comparable:

* the pinned LLVM-21 toolchain location and the exact pass pipelines,
* assert counting (which is stage-dependent -- see `count_asserts`),
* the compile / run / classify protocol,
* the §7 manifest oracle (`corrupts` vs `noop`),
* record emission against `schema/record-schema.json`.

Measurement rules that are easy to get wrong and are therefore enforced here:

1. **Assert counting depends on the IR stage.** `cf.assert` exists only at MLIR
   level; after full LLVM lowering it becomes `assert_msg_N` globals plus `abort`
   calls. Counting `cf.assert` in lowered IR silently returns 0. Use
   `count_asserts`, which picks the right pattern for the stage.
2. **`stdbuf -o0` is mandatory** when running: the assert message is written to
   stdout via `puts` before `abort()` and is otherwise lost to buffering.
3. **A missing shared library gives exit 1** (JIT symbol error), which is trivially
   misread as a detection. `run_kernel` distinguishes that case explicitly.
4. **RTV placement is a correctness requirement.** It instruments `memref.load` /
   `tensor.extract` but *not* `affine.load`, so `lower-affine` must run before it.
   The pipelines below encode the validated orderings.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --------------------------------------------------------------------------
# Toolchain (pinned in benchmark2/manifest.md §5.1)
# --------------------------------------------------------------------------

LLVM_ROOT = Path(
    os.environ.get("MLIR_LLVM_ROOT", "/home/garfee/dev/croqtile/extern/llvm-project")
)
LLVM_BIN = LLVM_ROOT / "bin"
LLVM_LIB = LLVM_ROOT / "lib"

MLIR_OPT = LLVM_BIN / "mlir-opt"
# `mlir-cpu-runner` was renamed `mlir-runner` in LLVM 21.
MLIR_RUNNER = LLVM_BIN / "mlir-runner"

RUNNER_LIBS = ",".join(
    str(LLVM_LIB / n) for n in ("libmlir_runner_utils.so", "libmlir_c_runner_utils.so")
)

TOOLCHAIN_VERSION = "LLVM 21.1.0"

# Exit codes we care about.
EXIT_ABORT = 134  # assert fired -> abort()
EXIT_SEGV = 139
EXIT_JIT_SYMBOL = 1  # missing shared libs / unresolved symbol -- NOT a detection
# Synthetic: the kernel never terminated within `timeout`. Distinct from every
# real exit status so a hang is classified on its own merits rather than falling
# into the generic "nonzero rc" bucket. An OOB write can corrupt the heap in a way
# that loops forever instead of faulting, so this is a genuine detection outcome
# for M1, not a harness failure.
EXIT_TIMEOUT = -1

_JIT_SYMBOL_MARKERS = (
    "JIT session error",
    "Symbols not found",
    "Failed to materialize symbols",
)

# --------------------------------------------------------------------------
# Pass pipelines (validated on LLVM 21.1.0)
# --------------------------------------------------------------------------
#
# Pass nesting matters: `finalize-memref-to-llvm`, `convert-cf-to-llvm`,
# `convert-func-to-llvm`, `convert-arith-to-llvm`, `convert-scf-to-cf`,
# `convert-vector-to-llvm`, `one-shot-bufferize` and
# `buffer-deallocation-pipeline` are module-level and must sit directly under
# `builtin.module(...)`. `lower-affine`, `convert-linalg-to-loops` and
# `generate-runtime-verification` are func-level.

# Tail shared by every pipeline: MLIR dialects -> LLVM.
#
# `convert-math-to-llvm` is required because the composed kernels use
# `math.exp` (softmax) and `math.sqrt` (layer_norm).
# `expand-strided-metadata` is required because `tensor.concat` bufferizes to
# `memref.subview` with a strided layout that `finalize-memref-to-llvm` will not
# convert on its own.
_TAIL = (
    "convert-scf-to-cf,"
    "convert-vector-to-llvm,"
    "convert-math-to-llvm,"
    "convert-index-to-llvm,"  # required: RTV emits index.bool.constant
    "expand-strided-metadata,"
    "convert-arith-to-llvm,"
    "finalize-memref-to-llvm,"
    "convert-cf-to-llvm,"
    "convert-func-to-llvm,"
    "reconcile-unrealized-casts"
)

# --- mlir-linalg: tensor-level linalg, needs bufferization -----------------
_LINALG_BODY_OFF = (
    "func.func(lower-affine),"
    "one-shot-bufferize,"
    "buffer-deallocation-pipeline,"
    "func.func(convert-linalg-to-loops)"
)
# RTV placement is NOT a free choice -- it is forced, and it scales S9 by ~4x, so
# the reasoning is recorded here rather than left implicit.
#
# Two candidate placements were measured on the relu kernel:
#
#   (A) bufferize -> convert-linalg-to-loops -> lower-affine -> RTV   [PINNED]
#       RTV sees memref.load/memref.store. 5 cf.assert, all
#       "^ out-of-bounds access": dynamic memory guards. Lowers to LLVM cleanly
#       and runs (exit 0).
#
#   (B) convert-linalg-to-loops -> RTV, with NO bufferization
#       20 cf.assert, but they are "^ dimension #N ... incompatible" and
#       "^ unexpected negative result on dimension #N": *static* shape checks,
#       not memory guards. convert-linalg-to-loops is a no-op on tensor ops, so
#       12 linalg ops survive to the LLVM tail, which then fails with
#       "Dialect `tensor' not found for custom op 'tensor.empty'". The kernel
#       cannot be built, let alone run.
#
# (B) is therefore not a valid configuration at all, and its higher count is not
# a real measurement of instrumentation cost. (A) is pinned because it is the only
# placement that yields an executable kernel. lower-affine must sit between
# convert-linalg-to-loops and RTV because linalg->loops emits affine.load.
_LINALG_BODY_ON = (
    "func.func(lower-affine),"
    "one-shot-bufferize,"
    "buffer-deallocation-pipeline,"
    "func.func(convert-linalg-to-loops,lower-affine,generate-runtime-verification)"
)

PIPELINES = {
    ("linalg", False): f"builtin.module({_LINALG_BODY_OFF},{_TAIL})",
    ("linalg", True): f"builtin.module({_LINALG_BODY_ON},{_TAIL})",
    # --- mlir-low: already on memref, no bufferization --------------------
    ("low", False): f"builtin.module(func.func(lower-affine),{_TAIL})",
    ("low", True): (
        "builtin.module(func.func(lower-affine,generate-runtime-verification,"
        f"canonicalize,cse),{_TAIL})"
    ),
}

# Pipeline used only to count how many checks RTV generated, before lowering.
RTV_ONLY = {
    "linalg": (
        "builtin.module(func.func(lower-affine),one-shot-bufferize,"
        "buffer-deallocation-pipeline,"
        "func.func(convert-linalg-to-loops,lower-affine,"
        "generate-runtime-verification))"
    ),
    "low": "builtin.module(func.func(lower-affine,generate-runtime-verification))",
}


def pipeline(surface: str, rtv: bool) -> str:
    """Return the pinned pass pipeline for a surface and RTV setting."""
    return PIPELINES[(surface, rtv)]


# --------------------------------------------------------------------------
# Process helpers
# --------------------------------------------------------------------------


@dataclass
class ProcResult:
    rc: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.rc == 0


def _as_text(v) -> str:
    """Normalize captured output to str.

    Even with `text=True`, `subprocess.TimeoutExpired` carries `stdout`/`stderr`
    as **bytes** -- a long-standing CPython quirk. Concatenating a str marker onto
    them raises `TypeError: can't concat str to bytes`, which would turn a hung
    kernel into a harness crash. Out-of-bounds mutants hang often enough (an OOB
    *write* corrupts the heap, and the damage may not surface until a later
    allocation) that this path is routinely hit, not theoretical.
    """
    if v is None:
        return ""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return v


def _run(cmd: list[str], timeout: int = 300) -> ProcResult:
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return ProcResult(p.returncode, p.stdout, p.stderr)
    except subprocess.TimeoutExpired as e:
        return ProcResult(
            EXIT_TIMEOUT,
            _as_text(e.stdout),
            _as_text(e.stderr) + f"\n[mlirbench] TIMEOUT after {timeout}s",
        )
    except FileNotFoundError as e:
        return ProcResult(-2, "", f"[mlirbench] binary not found: {e}")


def run_mlir_opt(pipeline_str: str, src: Path, dst: Path | None = None) -> ProcResult:
    """Run `mlir-opt` with an explicit pass pipeline.

    `dst=None` means "parse/verify only" (output discarded), which is how the
    verifier gate is measured.
    """
    cmd = [str(MLIR_OPT), f"--pass-pipeline={pipeline_str}", str(src)]
    if dst is not None:
        cmd += ["-o", str(dst)]
    else:
        cmd += ["-o", os.devnull]
    return _run(cmd)


def verify(src: Path) -> ProcResult:
    """Bare verifier gate: no passes, just parse + verify."""
    return _run([str(MLIR_OPT), str(src), "-o", os.devnull])


def run_kernel(
    ll_mlir: Path, entry_result: str = "i32", timeout: int = 120
) -> ProcResult:
    """Execute lowered IR with `mlir-runner`.

    `stdbuf -o0` is mandatory: the RTV assert message goes to stdout via `puts`
    before `abort()` and is lost to buffering otherwise.
    """
    cmd = [
        "stdbuf",
        "-o0",
        str(MLIR_RUNNER),
        f"--entry-point-result={entry_result}",
        f"--shared-libs={RUNNER_LIBS}",
        str(ll_mlir),
    ]
    return _run(cmd, timeout=timeout)


# --------------------------------------------------------------------------
# Assert counting (stage-dependent -- do not inline a grep)
# --------------------------------------------------------------------------

_CF_ASSERT = re.compile(r"\bcf\.assert\b")
_ASSERT_MSG_GLOBAL = re.compile(r"assert_msg_\d+")


def count_asserts(path: Path, lowered: bool) -> int:
    """Count runtime-verification checks in an IR file.

    `lowered=False` -> the file is still at MLIR level, count `cf.assert`.
    `lowered=True`  -> the file is fully lowered to LLVM; `cf.assert` no longer
                       exists and the checks appear as `assert_msg_N` globals.
                       Counting `cf.assert` here would silently return 0.
    """
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return 0
    if lowered:
        return len(set(_ASSERT_MSG_GLOBAL.findall(text)))
    return len(_CF_ASSERT.findall(text))


# RTV bakes the source location into each assert message. In MLIR text the quotes
# arrive octal-escaped -- `Location: loc(\22kernel\22)` -- while the same message
# decoded out of a lowered `assert_msg` global carries plain quotes,
# `Location: loc("kernel")`. Accept both spellings.
_LOC_TAG = re.compile(r'Location:\s*loc\((?:\\22|\\"|")?([A-Za-z_][\w.-]*)')


def count_asserts_by_loc(path: Path) -> dict[str, int]:
    """Attribute each `cf.assert` in RTV-only IR to its source location.

    A composed kernel contains two kinds of code: the operator under audit
    (`loc("kernel")`) and this harness's checksum oracle (`loc("oracle")`). RTV
    instruments *both*, because both lower to `memref.load`/`memref.store`. For
    S9 (remainder = Σ unconditional_guards) only the kernel's guards count -- the
    oracle's are harness artifacts, and folding them in inflates the remainder.
    Measured on relu: 3 kernel guards, 2 oracle guards, so reporting the raw total
    would have overstated S9 by 67% for that category.

    Returns a mapping of location name to guard count.
    """
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return {}
    counts: dict[str, int] = {}
    for line in text.splitlines():
        if "cf.assert" not in line:
            continue
        m = _LOC_TAG.search(line)
        key = m.group(1) if m else "<untagged>"
        counts[key] = counts.get(key, 0) + 1
    return counts


def count_kernel_asserts(path: Path) -> tuple[int, int]:
    """Return ``(kernel_guards, total_guards)`` from RTV-only IR.

    `kernel_guards` is the number to report for S9. Untagged asserts are counted
    as kernel guards, so a missing `loc` degrades to the over-counting behaviour
    rather than silently reporting zero.
    """
    counts = count_asserts_by_loc(path)
    total = sum(counts.values())
    return total - counts.get("oracle", 0), total


def assert_messages(path: Path) -> list[str]:
    """Decode the assert message payloads embedded in lowered IR.

    RTV stores each message in an `assert_msg_N` global as a `dense<"...">`
    tensor of i8. On LLVM 21 the payload is **hex-encoded** (`dense<"0x4552...">`,
    one byte per two hex digits, NUL-terminated); earlier builds used C escapes
    (`\\0A`, `\\22`). Both are handled, because mis-decoding silently yields
    garbage and would corrupt the bounds-vs-alignment distinction that settled the
    M3 x mlir-low question.
    """
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    out = []
    # The declaration is
    #   llvm.mlir.global private constant @assert_msg_3(dense<"0x4552..."> : tensor<189xi8>)
    # so the payload is terminated by `">`, not by `")`. Matching `")` finds
    # nothing and silently returns an empty list, which reads as "RTV emitted no
    # messages" rather than "the decoder is broken".
    for m in re.finditer(r'assert_msg_\d+\(dense<"(.*?)"\s*>', text, re.S):
        raw = m.group(1)
        decoded = _decode_payload(raw)
        if decoded is None:
            out.append(raw)
        else:
            out.append(decoded)
    return out


def _decode_payload(raw: str) -> str | None:
    """Decode one `assert_msg` payload; None if it is not a recognized form."""
    body = raw.replace("\\\n", "").replace("\n", "").strip()
    if body.startswith("0x") or body.startswith("0X"):
        hexstr = body[2:]
        if len(hexstr) % 2:
            return None
        try:
            data = bytes.fromhex(hexstr)
        except ValueError:
            return None
        # Trailing NUL terminator.
        return data.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    try:
        return body.encode("utf-8").decode("unicode_escape")
    except Exception:
        return None


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


@dataclass
class Classification:
    """One mutant's measured behaviour.

    `outcome` is the §6 taxonomy: compile / runtime / never / n/a.
    `manifest` is the §7 ground-truth oracle: corrupts / noop.
    """

    outcome: str = "never"
    stage: str = "compile"
    manifest: str = "noop"
    compile_ok: bool = False
    run_ok: bool = False
    ref_check: bool = False
    n_asserts: int = 0
    n_asserts_total: int = 0
    mismatch_count: int = -1
    abort_message: str = ""
    compile_error: str = ""
    notes: str = ""

    def as_record(self) -> dict:
        return asdict(self)


def classify(
    src: Path,
    work: Path,
    surface: str,
    rtv: bool,
    na: bool = False,
    na_reason: str = "",
    run_timeout: int = 60,
) -> Classification:
    """Compile, run and classify one kernel.

    The kernel's `@main` returns i32: `0` when the output's checksums (sum and
    sum-of-squares) both match the numpy reference baked in at compose time, `1`
    when either disagrees. That single value is both the ref-check gate and the
    §7 manifest oracle:

    * compile error                      -> outcome=compile,  manifest=corrupts
    * RTV assert fires (exit 134)        -> outcome=runtime, manifest=corrupts
    * kernel hangs (timeout)             -> outcome=runtime, manifest=corrupts
    * exit 0, mismatch > 0               -> outcome=never,    manifest=corrupts
    * exit 0, mismatch == 0              -> outcome=never,    manifest=noop
                                            (a false success; specs §7.1 discards it)

    Note the `never`/`corrupts` row is the interesting one for this lane: the
    defect survived the verifier *and* RTV, yet the oracle proves the output is
    wrong. That is precisely the silent-bug residue the paper measures.

    `run_timeout` is deliberately short. These kernels are tiny (small-input
    convention, manifest §5.2) and a correct one finishes in milliseconds, so a
    hang is a property of the injected defect rather than of the workload. The M1
    lane needs this more than M2: an out-of-bounds *write* corrupts the heap or
    wraps an index so a loop bound is never reached, and that surfaces as a hang
    instead of a fault.
    """
    if na:
        return Classification(
            outcome="n/a", stage="compile", manifest="noop", notes=na_reason
        )

    c = Classification()
    work.mkdir(parents=True, exist_ok=True)
    ll = work / f"{src.stem}.ll.mlir"

    # --- compile gate -----------------------------------------------------
    res = run_mlir_opt(pipeline(surface, rtv), src, ll)
    if not res.ok:
        c.compile_ok = False
        c.outcome = "compile"
        c.stage = "compile"
        c.compile_error = (res.stderr or res.stdout).strip()[:2000]
        # A mutant that fails to compile cannot be run, so the §7 oracle is
        # answered statically: the defect is real (it broke the build).
        c.manifest = "corrupts"
        return c
    c.compile_ok = True

    # How many checks did RTV generate? (feeds E3 remainder counting)
    #
    # `n_asserts` counts ONLY guards attributed to loc("kernel"). RTV also
    # instruments this harness's checksum oracle, whose loads/stores are equally
    # real memref accesses; those guards are harness artifacts and must not enter
    # S9. `n_asserts_total` keeps the raw count so the split stays auditable.
    rtv_ir = work / f"{src.stem}.rtv.mlir"
    if rtv:
        rres = run_mlir_opt(RTV_ONLY[surface], src, rtv_ir)
        if rres.ok:
            c.n_asserts, c.n_asserts_total = count_kernel_asserts(rtv_ir)

    # --- run gate ---------------------------------------------------------
    run = run_kernel(ll, entry_result="i32", timeout=run_timeout)
    return _verdict_from_run(run, run_timeout, c)


def _verdict_from_run(
    run: ProcResult, run_timeout: int, c: Classification | None = None
) -> Classification:
    """Map one `mlir-runner` result onto the §6 outcome / §7 manifest taxonomy.

    Factored out of `classify` so `classify_repeat` can apply the *identical*
    verdict logic to each of N runs of an already-lowered kernel. The two must
    not diverge: a verdict rule that lives in only one of them is exactly how the
    lanes drift out of comparability.
    """
    if c is None:
        c = Classification()
    combined = (run.stdout or "") + "\n" + (run.stderr or "")

    # Guard against misreading a JIT symbol error as a detection.
    if run.rc == EXIT_JIT_SYMBOL and any(m in combined for m in _JIT_SYMBOL_MARKERS):
        c.run_ok = False
        c.outcome = "never"
        c.manifest = "noop"
        c.notes = "JIT symbol error (missing shared libs) -- not a detection"
        c.abort_message = combined.strip()[:1000]
        return c

    if run.rc == EXIT_ABORT:
        c.run_ok = False
        c.outcome = "runtime"
        c.stage = "runtime"
        c.manifest = "corrupts"
        # First line of the RTV diagnostic, e.g. "^ out-of-bounds access".
        c.abort_message = _first_diagnostic(run.stdout) or combined.strip()[:1000]
        return c

    if run.rc == EXIT_TIMEOUT:
        # The kernel never terminated. This is a real detection outcome, not a
        # harness failure: an out-of-bounds *write* can corrupt the heap or an
        # index can wrap so the loop bound is never reached, and the damage shows
        # up as a hang rather than a fault. Kept distinct from `abort` so the
        # record says which mechanism caught the defect.
        c.run_ok = False
        c.outcome = "runtime"
        c.stage = "runtime"
        c.manifest = "corrupts"
        c.abort_message = f"TIMEOUT: kernel did not terminate within {run_timeout}s"
        return c

    if run.rc not in (0, EXIT_SEGV):
        c.run_ok = False
        c.outcome = "runtime"
        c.stage = "runtime"
        c.manifest = "corrupts"
        c.abort_message = f"exit {run.rc}: " + combined.strip()[:800]
        return c

    if run.rc == EXIT_SEGV:
        c.run_ok = False
        c.outcome = "runtime"
        c.stage = "runtime"
        c.manifest = "corrupts"
        c.abort_message = "SIGSEGV"
        return c

    # --- exit 0: read the mismatch count from stdout ----------------------
    c.run_ok = True
    c.mismatch_count = _parse_return(run.stdout)
    if c.mismatch_count < 0:
        c.notes = "could not parse runner return value"
        c.manifest = "noop"
        return c
    c.ref_check = c.mismatch_count == 0
    c.manifest = "noop" if c.ref_check else "corrupts"
    c.outcome = "never"
    c.stage = "runtime"
    return c


def classify_repeat(
    src: Path,
    work: Path,
    surface: str,
    rtv: bool,
    n: int = 5,
    na: bool = False,
    na_reason: str = "",
    run_timeout: int = 30,
) -> list[Classification]:
    """Compile once, run `n` times, return every verdict.

    WHY THIS EXISTS. A dropped-boundary-mask defect (M1.1) injects *undefined
    behavior*: the guardless out-of-bounds write may land idempotently (output
    unchanged -> `noop`), corrupt the heap (`abort`), or wrap an index so a loop
    bound is never reached (`hang`). Which happens depends on heap layout and
    varies run to run. Measured on `mlir-low` small-size: relu/static M1.1 at
    RTV-off gives ~3 `noop` / ~3 `abort` over 6 runs, and transpose/dyn M1.1
    gives ~5 `corrupts` / ~1 `noop`. A single `classify()` therefore *samples the
    UB once* and its verdict -- and any finding derived from it -- is not
    reproducible. This compiles the (deterministic) lowering once and runs it `n`
    times so the caller can aggregate over the distribution.

    Compilation is deterministic, so a compile failure is returned replicated to
    `n` identical verdicts; only the run gate is sampled.
    """
    if na:
        return [
            Classification(outcome="n/a", stage="compile", manifest="noop", notes=na_reason)
        ] * n

    work.mkdir(parents=True, exist_ok=True)
    ll = work / f"{src.stem}.ll.mlir"
    res = run_mlir_opt(pipeline(surface, rtv), src, ll)
    if not res.ok:
        c = Classification()
        c.compile_ok = False
        c.outcome = "compile"
        c.stage = "compile"
        c.compile_error = (res.stderr or res.stdout).strip()[:2000]
        c.manifest = "corrupts"
        return [c] * n

    n_asserts = n_asserts_total = 0
    if rtv:
        rtv_ir = work / f"{src.stem}.rtv.mlir"
        rres = run_mlir_opt(RTV_ONLY[surface], src, rtv_ir)
        if rres.ok:
            n_asserts, n_asserts_total = count_kernel_asserts(rtv_ir)

    out: list[Classification] = []
    for _ in range(n):
        run = run_kernel(ll, entry_result="i32", timeout=run_timeout)
        c = _verdict_from_run(run, run_timeout)
        c.compile_ok = True
        c.n_asserts = n_asserts
        c.n_asserts_total = n_asserts_total
        out.append(c)
    return out


def reduce_verdicts(verdicts: list[Classification]) -> tuple[Classification, dict]:
    """Collapse N runs of one (mutant, mode) into one canonical record + census.

    Canonical rule (conservative -- a defect that manifests in *any* run is not a
    false success), applied in order:

    1. every run `compile`            -> `compile`/`corrupts` (deterministic).
    2. any run `runtime` (abort/hang/segv/nonzero) -> `runtime`/`corrupts`.
       The defect faulted at least once, so the surface did not silently accept
       it. NOTE for S1: at RTV-off this fault is *incidental UB*, not a generated
       check -- see mlir-shared/README.md "UB-nondeterministic mutants".
    3. else all runs `never`: any `corrupts` -> `never`/`corrupts` (output wrong
       in at least one run); only if *every* run is `noop` -> `never`/`noop`.

    A mutant is therefore a `noop` false success (specs §7.1) only when it is
    noop in *every* run of *every* mode -- the correct reading of "did this
    mutant actually corrupt anything?" under undefined behavior.

    Returns `(canonical, census)` where `census` maps `(outcome, manifest)` to
    its run count, so the caller can report the distribution rather than a
    single sampled verdict.
    """
    census: dict[tuple[str, str], int] = {}
    for v in verdicts:
        census[(v.outcome, v.manifest or "-")] = census.get((v.outcome, v.manifest or "-"), 0) + 1

    if all(v.outcome == "compile" for v in verdicts):
        return verdicts[0], census
    if any(v.outcome == "runtime" for v in verdicts):
        # Pick a representative runtime verdict (prefer one carrying a diagnostic).
        rep = next((v for v in verdicts if v.outcome == "runtime" and v.abort_message), None)
        rep = rep or next(v for v in verdicts if v.outcome == "runtime")
        return rep, census
    # all `never` (or n/a): corrupts wins over noop unless every run is noop.
    if any(v.manifest == "corrupts" for v in verdicts):
        rep = next(v for v in verdicts if v.manifest == "corrupts")
        return rep, census
    return verdicts[0], census


def _first_diagnostic(stdout: str) -> str:
    """Pull the human-readable RTV diagnostic out of runner stdout."""
    for line in (stdout or "").splitlines():
        s = line.strip()
        if s.startswith("^") or "verification failed" in s or "Location:" in s:
            return s
    return ""


def _parse_return(stdout: str) -> int:
    """`mlir-runner --entry-point-result=i32` prints the value on its own line."""
    for line in reversed((stdout or "").strip().splitlines()):
        s = line.strip()
        if re.fullmatch(r"-?\d+", s):
            return int(s)
    return -1


# --------------------------------------------------------------------------
# Provenance / hashing (non-negotiable cross-cutting fields)
# --------------------------------------------------------------------------


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def short(h: str, n: int = 12) -> str:
    return h[:n]


def settings_hash(category: str, benchmark2: Path) -> str:
    """Hash of the shared settings file for a category (provenance only)."""
    p = benchmark2 / "settings" / f"{category}.md"
    return short(sha1_file(p)) if p.exists() else "MISSING"


# --------------------------------------------------------------------------
# Record emission (schema/record-schema.json)
# --------------------------------------------------------------------------

ENUMS = {
    "size": {"small", "full"},
    "exclusive": {"true", "false"},
    "compile": {"ok", "fail", "detected"},
    "run": {"ok", "crash"},
    "ref_check": {"pass", "fail"},
    "class": {"M1", "M2", "M3"},
    "paper_category": {"dim-mismatch", "oob", "wrong-shape", "stride"},
    "level": {"1", "2"},
    "outcome": {"compile", "runtime", "never", "n/a"},
    "stage": {"compile", "runtime"},
    "manifest": {"corrupts", "noop"},
    "expressible": {"yes", "no", "partial"},
}


def validate(record: dict, kind: str, schema: dict) -> list[str]:
    """Validate a record against schema/record-schema.json. Returns errors."""
    errs = []
    spec = schema.get("records", {}).get(kind)
    if spec is None:
        return [f"unknown record kind {kind!r}"]
    for f in spec.get("fields", []):
        if f not in record:
            errs.append(f"{kind}: missing field {f!r}")
    for f, allowed in spec.get("enums", {}).items():
        if f in record and str(record[f]) not in set(allowed):
            errs.append(
                f"{kind}: field {f!r}={record[f]!r} not in {sorted(allowed)}"
            )
    return errs


def load_schema(benchmark2: Path) -> dict:
    return json.loads((benchmark2 / "schema" / "record-schema.json").read_text())


class RecordWriter:
    """Appends JSON-lines to `raw/<stream>.jsonl` (one object per observation)."""

    def __init__(self, raw_dir: Path, stream: str, schema: dict, kind: str):
        self.path = raw_dir / f"{stream}.jsonl"
        self.schema = schema
        self.kind = kind
        self.errors: list[str] = []
        raw_dir.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        errs = validate(record, self.kind, self.schema)
        if errs:
            self.errors.extend(errs)
            print(f"[mlirbench] SCHEMA ERROR: {errs}", file=sys.stderr)
        with self.path.open("a") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def truncate(self) -> None:
        """Clear the stream (idempotent re-runs)."""
        self.path.write_text("")


# --------------------------------------------------------------------------
# Environment self-check
# --------------------------------------------------------------------------


def check_toolchain() -> list[str]:
    """Return a list of problems with the pinned toolchain (empty == healthy)."""
    problems = []
    for b in (MLIR_OPT, MLIR_RUNNER):
        if not b.exists():
            problems.append(f"missing binary: {b}")
    for lib in RUNNER_LIBS.split(","):
        if not Path(lib).exists():
            problems.append(f"missing runner lib: {lib}")
    if not problems:
        v = _run([str(MLIR_OPT), "--version"])
        m = re.search(r"LLVM version (\S+)", v.stdout)
        if m and m.group(1) != "21.1.0":
            problems.append(
                f"expected LLVM 21.1.0, found {m.group(1)} at {MLIR_OPT}"
            )
    return problems


def describe_toolchain() -> str:
    """One-line human-readable toolchain status for validator output.

    `check_toolchain()` returns a list of problems, which prints as `[]` when
    healthy and tells the reader nothing about what was actually checked. Every
    record carries the toolchain version, so the resolved paths belong in the log
    too -- a number is only reproducible if the binary that produced it is named.
    """
    problems = check_toolchain()
    head = f"{TOOLCHAIN_VERSION}  mlir-opt={MLIR_OPT}"
    if problems:
        return head + "\n  TOOLCHAIN PROBLEMS:\n  - " + "\n  - ".join(problems)
    return head + "  [ok]"
