#!/usr/bin/env python3
"""E5 — runtime detection cost & latency (choreo only). Feeds S13 → RQ3 + E5.

Plan §6.2. TWO sub-lanes, both required, both TIMING lanes.

==========================================================================
PHASE-0 GATE: SATISFIED (plan §6.2 "Phase-0 gate")
==========================================================================
The plan says the coordinator "must confirm the `--runtime-checks off` flag
exists and pin it. If it does not, E5a is unmeasurable." It exists. Manifest §3
maps the plan's spelling to the real one:

    --runtime-checks off   →   -rtc=none
    --runtime-checks on    →   choreo's DEFAULT (no -rtc flag)

Verified empirically on layer_normalization/3_attention_32xNx512x64_64_64 by
diffing the generated CUDA with the scratch-dir path normalized out:

    -rtc level   runtime_check(   choreo_assert(   differs from DEFAULT
    DEFAULT                 6                1     —
    entry                   6                1     0 lines   (== DEFAULT)
    low                     6                1     0 lines   (== DEFAULT)
    medium                  6                5     4 lines
    high                    6                5     4 lines
    all                     6                5     4 lines
    none                    0                1     7 lines

So DEFAULT == entry == low, byte-identical. `-rtc=none` removes all 6 entry
guards and is a clean checks-off arm. `medium`+ additionally emits 4 *interior*
`choreo_assert` guards inside the loop body — which is exactly the entry-vs-
interior distinction E5b must record as a MEASURED field.

==========================================================================
E5a — per-run residue overhead (the honest RQ3)
==========================================================================
Same kernel, compiled twice (checks on / checks off), executed, Δ taken.
Scoped to DYNAMIC-shape cases: plan §6 says static cases discharge 100% at
compile time. That premise is tested here rather than assumed — static cases are
measured too when --include-static is given, because E2/E3 found static kernels
DO emit entry shape guards (see the closing report).

Timing source, in priority order:
  1. the kernel's own `Execution time: N us|microseconds` marker — brackets the
     choreo call, excluding process startup. Present in all 15 categories
     (8 carry it in their common.h/.hpp rather than the .co).
  2. total process wall-clock — used only when no marker exists; contaminated by
     ~constant process startup, so it understates the relative Δ. The field
     `timing_source` records which was used so the integrator can tell.

==========================================================================
E5b — detection latency vs the dynamic oracle
==========================================================================
On E1's runtime-outcome mutants, time-from-launch-to-report for:
  (i)  choreo's own entry check        detector=choreo-entry
  (ii) compute-sanitizer over choreo's generated CUDA built with -rtc=none, so
       the sanitizer is the SOLE detector, not racing choreo's check
                                        detector=compute-sanitizer

`check_loc` is MEASURED, not assumed: the arm whose report comes from a
`runtime_check` at function entry is `entry`; one from a `choreo_assert` inside
a loop body is `interior`.

Time-to-report is subprocess wall-clock from exec to the detector's message.
Both arms pay the same fixed process-startup cost, so the COMPARISON is clean
even though the absolute numbers carry that constant. Stated on every record.

FAIRNESS (plan §6.2, do not strawman): compute-sanitizer is a debugger, not a
fast checker. The claim is structural — hoisted entry checks vs run-to-fault
instrumentation are different latency MODELS — not "sanitizer is slow". Its
per-run slowdown is folded into the latency number, not quoted separately.

==========================================================================
EXCLUSIVITY
==========================================================================
Both sub-lanes touch the device, so both stamp `exclusive=true` and must run
under `benchmark2/.gpu-lock` (run.sh:cmd_e5 wraps this in `exclusive`). qc
rejects residue/latency records with exclusive=false. This script CHECKS for the
lock and warns loudly if it is absent rather than silently claiming exclusivity.

FEEDS  S13 (E5a Δ per dynamic-shape case; E5b choreo-entry vs sanitizer).
"""

import argparse
import collections
import json
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))          # benchmark2/choreo
B2 = os.path.dirname(HERE)                                 # benchmark2/
REPO = os.path.dirname(B2)                                 # svn-artifacts/

sys.path.insert(0, HERE)
import run_e2                                               # noqa: E402
import gpuinfo                                              # noqa: E402
import toolchain                                            # noqa: E402

RAW = os.path.join(HERE, "raw")
OUT = os.path.join(RAW, "e5_runtime.json")
WORKDIR = os.path.join(RAW, "e5_logs")
E2_LEDGER = os.path.join(RAW, "e2_ledger.json")
E1_RECORDS = os.path.join(RAW, "e1_mutant_records.json")
MANIFEST = os.path.join(RAW, "mutant_manifest.json")

TOOLCHAIN = "choreo"
CHOREO = os.path.join(REPO, "croqtile", "build-release", "choreo")
SUITE = os.path.join(REPO, "benchmark", "choreo")
MUTANTS = os.path.join(HERE, "mutants")
LOCK = os.path.join(B2, ".gpu-lock")

SANITIZER = "/usr/local/cuda/bin/compute-sanitizer"

CAP = "--max-local-mem-capacity=2000000"
BASE_FLAGS = ["-gs", "-t", "cute", "-kt", CAP]
# checks ON  = choreo's default (== -rtc=entry == -rtc=low, measured above)
FLAGS_ON = BASE_FLAGS
# checks OFF = the plan's `--runtime-checks off` (manifest §3).
#
# WHICH SPELLING, AND WHY THIS ONE. Three flags suppress the guards:
# `-rtc=none`, `--disable-runtime-check`, `-zero-cost`. Measured on
# layer_normalization/3_attention (generated .cu, path-normalized):
#
#     flag                      runtime_check(  choreo_assert(  bytes  diff vs default
#     default                                6               1   8360   —
#     -rtc=none                              0               1   7360   8 lines
#     --disable-runtime-check                0               1   7360   8 lines
#     -zero-cost                             0               1   7227  13 lines
#
# `-rtc=none` and `--disable-runtime-check` are BYTE-IDENTICAL (0 differing
# lines). `-zero-cost` removes 133 more bytes because it also drops the CUDA
# runtime environment check — a second, unrelated suppression that would make
# the A/B differ by more than the guards under test.
#
# The owner pinned `--disable-runtime-check` for exactly that reason, so that is
# what this lane uses. The equivalence above is recorded so a reviewer who sees
# `-rtc=none` elsewhere (the manifest's mapping, the other worker's
# scripts/choreo_runtime_entry.py) knows the two are the same build.
FLAGS_OFF = BASE_FLAGS + ["--disable-runtime-check"]
# E5b arm (ii): sanitizer must be the SOLE detector, so build checks-off.
FLAGS_SANITIZER = FLAGS_OFF

T_COMPILE = 1800
T_EXECUTE = 900

# The kernel's own timing marker. Not anchored: softmax/matmul print
# "Case <name> Execution time: N microseconds", others "Execution time: N us".
RE_EXECTIME = re.compile(r"Execution time:\s*(\d+)\s*(?:us|microseconds)", re.I)
RE_RTC = re.compile(r"choreo runtime check failed")
RE_ASSERT = re.compile(r"choreo assertion failed")
# choreo appends the source location to guards IT emits (`ar.message << ", " <<
# ar.loc`), rendering as ", <path>:<line>.<col>" — DOT before the column. The
# runtime-lib bounds check and the kernel's own oracle assert do NOT append one,
# so this suffix is what separates a choreo check from someone else's. See
# check_loc_of().
RE_LOC_SUFFIX = re.compile(r"\.(?:co|cu|hpp|h|cpp):[0-9]+\.[0-9]+")
RE_SANITIZER = re.compile(
    r"========= (Invalid|Program hit|ERROR|Barrier|Initization|Leaked)", re.I)
# compute-sanitizer's closing line is "========= ERROR SUMMARY: N errors". With
# N == 0 it is the tool saying it found NOTHING -- but the bare `ERROR`
# alternative in RE_SANITIZER matches it, so `flagged` came out True on arms
# that made no detection at all. See sanitizer_verdict(); RE_SANITIZER is kept
# as-is (it still selects the first report LINE) and the verdict is what decides.
RE_SANITIZER_SUMMARY = re.compile(
    r"ERROR SUMMARY:\s*(\d+)\s*error", re.I)
RE_SANITIZER_FAULT = re.compile(
    r"Invalid __global__|Invalid __shared__|Invalid __local__"
    r"|out-of-bounds|misaligned|Racecheck|Barrier error", re.I)
RE_SANITIZER_ABORT = re.compile(
    r"process didn't terminate successfully", re.I)

E5A_DEFINITION = (
    "residue A/B: same kernel compiled twice, checks-on = choreo default "
    "(measured byte-identical to -rtc=entry and -rtc=low) vs checks-off = "
    "--disable-runtime-check (the plan's `--runtime-checks off`, manifest §3; "
    "byte-identical to -rtc=none, and preferred over -zero-cost which also "
    "drops the unrelated CUDA env check). runtime_ms from the kernel's own "
    "`Execution time` marker where present, else process wall-clock; see "
    "timing_source."
)
E5B_DEFINITION = (
    "time_to_report_us = subprocess wall-clock from EXECUTING THE COMPILED "
    "BINARY to the detector's report on its stream. The nvcc compile is "
    "performed untimed, outside the measured region, in BOTH arms -- see the "
    "SYMMETRY FIX note in e5b_one. Both arms carry the same fixed "
    "process-startup constant, so the comparison is clean though the absolute "
    "is not. choreo-entry arm built with default -rtc; compute-sanitizer arm "
    "built with --disable-runtime-check so the sanitizer is the sole detector."
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def lock_held():
    """Is the GPU lock present? Stamping exclusive=true without it is a lie.

    Delegates to gpuinfo so E4 and E5 share ONE definition of the lock path and
    of what `exclusive` means. Two copies drift, and a timing lane that
    disagrees with the compile lane about exclusivity produces records qc
    judges by different rules.
    """
    return gpuinfo.lock_held()


def host_busy():
    """Competing compile/run processes on this host (see gpuinfo.host_busy).

    WHY THIS EXISTS. A first E5a cut reported checks-on 25% FASTER than
    checks-off, which is not a result — it was a concurrent E4 run saturating
    every core with nvcc. A timing lane cannot distinguish "the guard is free"
    from "the host was stolen" after the fact, so the contamination is detected
    up front and reported rather than discovered in the numbers.
    """
    return gpuinfo.host_busy()


def wait_for_quiet(timeout_s, poll_s=10, quiet_streak=3):
    """Block until `host_busy()` is empty, or give up after `timeout_s`.

    WHY THIS EXISTS. The original check was a single instantaneous sample, so a
    transient nvcc burst aborted a ~2 h unattended run. Observed 2026-09-09:
    `./run.sh e5` died with rc=3 on `['cicc', 'nvcc']` that were gone 40 s later
    (another user's remote e2e job does local compile steps). Polling six times
    over 30 s afterwards showed the host quiet the whole time.

    Returns (busy, waited_s). `busy` empty means quiet. A streak is required
    rather than one clean sample because nvcc forks cicc/ptxas in bursts, and a
    single gap between two bursts is not a quiet host.

    This must still run BEFORE any nvcc of our own is spawned: `host_busy()` uses
    `ps -eo comm=` and cannot tell our children from someone else's (see
    gpuinfo.host_busy's LIMITATION note).
    """
    t0 = time.time()
    streak = 0
    last = []
    while True:
        last = host_busy()
        if not last:
            streak += 1
            if streak >= quiet_streak:
                return [], time.time() - t0
        else:
            streak = 0
        if time.time() - t0 >= timeout_s:
            return last, time.time() - t0
        time.sleep(poll_s)


def prep(category, case, workdir, src_override=None):
    """Copy kernel + local headers into a scratch dir. Returns (dir, error).

    The generated CUDA resolves `#include "common.h"` via `-I<dir of the .co>`,
    so local headers MUST sit next to the copy or nvcc dies with "fatal error:
    common.h: No such file or directory".
    """
    d = os.path.join(workdir, f"{category}_{case}")
    os.makedirs(d, exist_ok=True)
    src = src_override or os.path.join(SUITE, category, case + ".co")
    if not os.path.exists(src):
        return None, f"source missing: {src}"
    with open(src, "rb") as f:
        blob = f.read()
    with open(os.path.join(d, "k.co"), "wb") as f:
        f.write(blob)
    catdir = os.path.join(SUITE, category)
    if os.path.isdir(catdir):
        for h in sorted(os.listdir(catdir)):
            if h.endswith((".h", ".hpp")):
                with open(os.path.join(catdir, h), "rb") as f:
                    hb = f.read()
                with open(os.path.join(d, h), "wb") as f:
                    f.write(hb)
    return d, None


def build(d, flags, tag):
    """Generate the script for one arm. Returns (script_path, error)."""
    out = os.path.join(d, f"k_{tag}.sh")
    r = subprocess.run([CHOREO] + flags + [os.path.join(d, "k.co"), "-o", out],
                       capture_output=True, timeout=T_COMPILE, cwd=REPO)
    if r.returncode != 0 or not os.path.exists(out) or os.path.getsize(out) == 0:
        return None, (f"choreo rc={r.returncode}: "
                      + r.stderr.decode("utf8", "replace")[-300:])
    return out, None


def exe_path(script):
    """The .exe the generated script builds. choreo deletes the .cu on exit but
    the script text still names both paths."""
    with open(script, "rb") as f:
        txt = f.read().decode("utf8", "replace")
    m = re.search(r"/tmp/[0-9_]+/__choreo_cute_k\.exe", txt)
    return m.group(0) if m else None


def run_exe(script, env_extra=None, timeout=T_EXECUTE):
    """Execute via the script's own --execute path. Returns (rc, stdout, stderr,
    wall_s).

    `stdbuf -o0 -e0` is MANDATORY: abort() discards buffered stdout, and the
    attribution message is printed on stdout immediately before the abort.
    Without it the detector/oracle messages vanish and every verdict is void.
    """
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    t = time.perf_counter()
    try:
        p = subprocess.run(["stdbuf", "-o0", "-e0", "bash", script, "--execute"],
                           capture_output=True, timeout=timeout, cwd=REPO, env=env)
        rc = p.returncode
        so = p.stdout.decode("utf8", "replace")
        se = p.stderr.decode("utf8", "replace")
    except subprocess.TimeoutExpired:
        rc, so, se = -1, "", "TIMEOUT"
    return rc, so, se, time.perf_counter() - t


def compile_then_run(d, script, env_extra=None):
    """--compile-link, then run the produced .exe directly.

    Needed for E5b's sanitizer arm: compute-sanitizer must wrap the EXECUTABLE,
    not the script. Returns (rc, stdout, stderr, wall_s, exe, error).
    """
    p = subprocess.run(["bash", script, "--compile-link"],
                       capture_output=True, timeout=T_COMPILE, cwd=REPO)
    if p.returncode != 0:
        return None, "", p.stderr.decode("utf8", "replace"), 0.0, None, \
            f"compile-link rc={p.returncode}"
    exe = exe_path(script)
    if not exe or not os.path.exists(exe):
        return None, "", "", 0.0, None, f"exe not found at {exe}"
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    t = time.perf_counter()
    try:
        q = subprocess.run(["stdbuf", "-o0", "-e0", exe],
                           capture_output=True, timeout=T_EXECUTE, cwd=REPO, env=env)
        rc, so, se = q.returncode, q.stdout.decode("utf8", "replace"), \
            q.stderr.decode("utf8", "replace")
    except subprocess.TimeoutExpired:
        rc, so, se = -1, "", "TIMEOUT"
    return rc, so, se, time.perf_counter() - t, exe, None


def timing_ms(stdout, wall_s):
    """(runtime_ms, timing_source). Prefer the kernel's own marker."""
    vals = [int(m.group(1)) for m in RE_EXECTIME.finditer(stdout)]
    if vals:
        # A kernel may print several cases; the residue Δ is per-run, so take the
        # sum — that is the total device work the run performed.
        return sum(vals) / 1000.0, "kernel-marker"
    return wall_s * 1000.0, "process-wallclock"


# --------------------------------------------------------------------------
# LAUNCH GUARD — "did it actually launch?"
# --------------------------------------------------------------------------
# WHY THIS EXISTS. croqtile commit 1fa4719 ("codegen: check launch errors after
# plain <<<>>> kernel launches") documents a failure mode that is invisible to
# every check this lane used to make:
#
#   "A rejected kernel launch only records the failure in the thread-local CUDA
#    error state; it does not raise anything at the call site. The generated host
#    code immediately followed the plain <<<...>>> launch with
#    cudaDeviceSynchronize(), which succeeds when nothing was enqueued, so the
#    kernel silently never ran: the output buffer kept its initial contents, the
#    harness still printed an 'Execution time', and the process exited 0."
#
# The trigger is this lane's own pinned flag. `--max-local-mem-capacity=2000000`
# makes the dynamic-shape path size the per-thread local arena to the full
# capacity (lib/mem_reuse.cpp:430-434); backing every RESIDENT thread then needs
# capacity * maxThreadsPerSM * numSMs = 466.9 GB on an H800 PCIe (114 SMs x 2048)
# against 85 GB of device memory, so cudaLaunchKernel returns
# cudaErrorInvalidValue. The driver reserves against capacity, NOT against the
# actual launch dims -- so even E5a's degenerate 2-thread launch is rejected.
# 36 of the 153 dynamic benchmark cases were affected and ALL reported success.
#
# MEASURED BLAST RADIUS on this lane's data (diag_launch_reject.py, all 14 E5a
# kernels + all 14 mutant bases, rebuilt on the post-fix binary):
#   E5a:  3 of 14 VOID -- matmul/10_dynamic_128x1280_1280xN_128xN,
#         max_pool2d/10_dynamic_32xCxHxW_32xCxHd5xWd5,
#         softmax/10_dynamic_16x512xHxW_16x512xHxW.
#         All three were recorded ok=True with 324-493 ms timings and
#         delta_within_noise=True: two no-ops compared to each other.
#   E5b:  2 of 16 pairs CONFOUNDED -- M2.s1.mm11.rhs (base matmul/11_dynamic)
#         and M3.s3.cv1.local8 (base conv2d/1_attention). The choreo-entry arm
#         is clean on all 16; the SANITIZER arm timed a rejected launch, which
#         depresses the ratio in exactly the confounded direction (1.62x and
#         2.32x against a clean median of 143.0x).
#   E1:   32 of 120 mutants sit on a rejected base (18 never / 12 compile /
#         2 runtime). The 2 runtime ones are exactly the E5b confounded pair,
#         and E1 independently recorded oracle_run_rc=0 + oracle_caught=none on
#         both -- the pre-fix silent-rejection signature.
#
# A timing lane cannot detect this from its own numbers: a rejected launch still
# prints a plausible millisecond value. So the launch is asserted directly.
RE_LAUNCH_REJECT = re.compile(
    r"cudaErrorInvalidValue"
    r"|CUDA failure:\s*1\b"
    r"|invalid argument"
    r"|cudaLaunchKernel"
    r"|launch (?:error|failed)"
    r"|abend",
    re.I)
RE_KERNEL_END = re.compile(r"Execution time", re.I)


def launch_verdict(rc, stdout, stderr):
    """(launch_ok, evidence). Did the kernel ACTUALLY run on the device?

    Three distinct failure classes, deliberately not collapsed into one:

      launch-rejected      the launch API refused. Post-1fa4719 this aborts
                           (rc=134) and names cudaLaunchKernel; pre-1fa4719 it
                           was silent, which is why the marker check below is
                           not sufficient on its own.
      no-kernel-end-marker rc==0 but the harness never printed "Execution time".
                           This is the "reported success while never launching"
                           class -- a false success, and the one that silently
                           depresses a median.
      nonzero-rc           failed for some other reason (timeout, abort from a
                           real check, segfault). Not a launch question.

    `launch_ok` is True ONLY when rc==0 AND the end marker is present AND no
    rejection text appears. Anything else is excluded from timing rather than
    averaged in.
    """
    blob = (stderr or "") + "\n" + (stdout or "")
    marker = bool(RE_KERNEL_END.search(blob))
    hit = None
    for line in blob.splitlines():
        if RE_LAUNCH_REJECT.search(line):
            hit = line.strip()[:240]
            break
    if hit:
        reason = "launch-rejected"
    elif rc != 0:
        reason = "nonzero-rc"
    elif not marker:
        reason = "no-kernel-end-marker"
    else:
        reason = None
    ev = {
        "rc": rc,
        "kernel_end_marker": marker,
        "rejection_text": hit,
        "reason": reason,
    }
    return reason is None, ev


def sanitizer_verdict(report, report_line_seen, baseline_rc, arm_rc=None):
    """(verdict, detected). What did compute-sanitizer ACTUALLY report?

    WHY THIS EXISTS. The second argument used to be called `flagged` and was
    `bool(RE_SANITIZER.search(...))`, and RE_SANITIZER's bare `ERROR`
    alternative matches compute-sanitizer's closing line
    "========= ERROR SUMMARY: 0 errors" -- the tool saying it found NOTHING. So
    4 of the 16 E5b arms were recorded flagged=True with no detection behind
    them, and because those 4 are the SLOWEST arms (68-162 s of instrumented
    execution that never faulted) they supplied the four largest ratios in the
    set: 162x, 190x, 215x and the max 472x. The headline "median 64x / max
    472x" was therefore carried by arms where the oracle made no report at all.
    E1 corroborates independently: 3 of those 4 have oracle_caught=none,
    oracle_run_rc=0, oracle_passed=True.

    Renamed to `report_line_seen` because the old name asserted an
    interpretation ("flagged as faulty") that the value never carried. It means
    only "the tool printed a line with a report prefix". The interpretation is
    this function's job and comes out as `verdict`/`detected`.

    `arm_rc` is the SANITIZER arm's own exit code (not the baseline's). It is
    what separates two verdicts that look identical from the report text alone:
      arm_rc == 0   the body ran to completion and nothing was found. That is
                    ORACLE BLINDNESS -- a coverage result.
      arm_rc != 0   the arm died before reporting. That is NOT a detection
                    judgement about the fault; the oracle never finished.
    The detection rule is "exit with no fault report AND rc != 0 -> confounded",
    so without arm_rc the rule cannot be evaluated at all. arm_rc=None means the
    record predates this field; the verdict then degrades to the report-only
    classification rather than guessing an exit code.

    Verdicts, in the order they are tested:
      launch-rejected    the launch API refused (see launch_verdict). The arm
                         timed a rejection, not the instrumented body.
      memcheck-fault     a real access violation. This IS a detection.
      no-fault-reported  "ERROR SUMMARY: 0 errors", no fault line, and (when
                         known) arm_rc == 0. The oracle ran the body to
                         completion and found nothing. No time-to-report exists.
      died-no-report     no fault line and arm_rc != 0. The arm exited without
                         reporting. Confounded, but NOT oracle blindness.
      process-abort      the wrapped process died. NOT a detection: measured on
                         all 6 such arms, the killer is choreo's own UNGATED
                         assertion (choreo.h:221 choreo_assert / choreo.h:886
                         ArrayProxy::operator[]) and memcheck reports zero
                         errors. See the branch for the evidence.
      no-report          nothing recognisable at all.
    """
    rep = report or ""
    if RE_LAUNCH_REJECT.search(rep):
        return "launch-rejected", False
    if RE_SANITIZER_FAULT.search(rep):
        return "memcheck-fault", True
    m = RE_SANITIZER_SUMMARY.search(rep)
    if m and int(m.group(1)) == 0 and not RE_SANITIZER_ABORT.search(rep):
        # Zero errors. arm_rc decides WHICH zero-error story this is.
        if arm_rc is not None and arm_rc != 0:
            return "died-no-report", False
        return "no-fault-reported", False
    if RE_SANITIZER_ABORT.search(rep):
        # detected=False, and this was the last silent inflation in E5b.
        #
        # The verdict name is right -- a process died -- but "a process died" is
        # NOT "compute-sanitizer detected the fault", and `detected` means the
        # latter. An earlier cut returned True here on the guess that the abort
        # was "typically the kernel's own oracle assert aborting". Measured, it
        # is: diag_abort_attribution.py ran all 6 such arms twice, with and
        # without compute-sanitizer. Every one aborts ALONE (baseline_rc=-6) at
        # choreo.h:221 (`choreo_assert`, source A-device) or choreo.h:886
        # (`ArrayProxy::operator[]`, source B), and under memcheck the tool's own
        # summary is "ERROR SUMMARY: 0 errors" on all 6.
        #
        # So the killer is choreo, not the oracle -- and note that BOTH candidate
        # killers are choreo's: source B is choreo's bounds check, and source C
        # (the user `assert` BIF) is choreo's own oracle mechanism. In E5b the
        # "dynamic oracle" IS compute-sanitizer, so an abort on any choreo
        # assertion is choreo detecting, never compute-sanitizer detecting.
        # Returning True here credited the oracle with choreo's kill and admitted
        # 6 arms to the latency ratio whose wall clock measured choreo -- i.e. a
        # ratio comparing choreo against choreo. Those 6 span 1.57x-233x, so the
        # bias is not even uniform.
        #
        # This also falsifies E5b's stated premise that --disable-runtime-check
        # leaves "the sanitizer as sole detector": the flag gates only source A,
        # while B and C are ungated by construction. Reported, not designed
        # around -- see stats.py's detection_asymmetry.premise_violation_note.
        return "process-abort", False
    if m and int(m.group(1)) > 0:
        return "memcheck-fault", True
    if report_line_seen:
        # RE_SANITIZER matched something this function does not classify. Do not
        # silently call it a detection: name it so a reader can look.
        return "unclassified-but-flagged", True
    if arm_rc is not None and arm_rc != 0:
        return "died-no-report", False
    return "no-report", False


def dynamic_kernels(include_static=False):
    """(category, case, shape_class) from the E2 ledger — the authoritative
    static/dynamic classification, not a re-derivation.

    E2 marks success with `compile == "ok"` (there is no `ok` field), and
    `kernel_id` is `category/case`. Reading either wrong silently yields an
    empty selection, which is why this is spelled out.
    """
    if not os.path.exists(E2_LEDGER):
        return None
    with open(E2_LEDGER) as f:
        led = json.load(f)
    out = []
    for k in led.get("kernels", []):
        if k.get("compile") != "ok":
            continue
        sc = k.get("shape_class")
        if sc == "dynamic" or (include_static and sc == "static"):
            kid = k["kernel_id"]
            case = kid.split("/", 1)[1] if "/" in kid else kid
            out.append((k["category"], case, sc))
    return out


# --------------------------------------------------------------------------
# E5a
# --------------------------------------------------------------------------

def device_code_evidence(scripts):
    """Prove WHERE the gated checks live, so a below-noise Δ is interpretable.

    A Δ inside the arms' own spread is ambiguous on its own: it could mean "no
    cost" or "cost too small to resolve against this host's clock ramp". This
    settles it structurally, from the two scripts E5a already built.

    Extracts each arm's `__global__` body from the generated CUDA (the heredoc
    `cat <<'EOF' > .../__choreo_cute_k.cu`) and reports:
      * `device_identical`  — the two device kernels are byte-identical, so
        `--disable-runtime-check` changes NOTHING that runs on the GPU. Any Δ is
        host-side only.
      * `device_check_calls` — `runtime_check(`/`choreo_assert(` count inside the
        device body. 0 means there are no device-side checks to cost anything.
      * `host_gated_checks` — gated `runtime_check` calls in the host wrapper,
        i.e. the thing the flag actually removes.

    Measured on layer_normalization/10_dynamic: 12 host-side gated checks
    (6 shape-compatibility + 6 zero-dim, all integer comparisons on shape
    metadata) at lines 8158-8170, kernel launch at 8185, timer at 8207 — so all
    12 run ONCE per launch, before the launch, and the device body (8077-8152)
    is identical between arms. That makes a sub-millisecond Δ the expected
    result, not a measurement failure.

    Returns {} rather than raising: this is evidence, not a precondition, and a
    parsing surprise must not lose a timing sample.
    """
    try:
        bodies, gated = {}, {}
        for tag, script in scripts.items():
            with open(script, "r", errors="replace") as f:
                lines = f.read().splitlines()
            body, in_body = [], False
            n_host = 0
            for ln in lines:
                if ln.startswith("__global__ "):
                    in_body = True
                if in_body:
                    body.append(ln)
                    if ln.startswith("}"):
                        in_body = False
                elif "runtime_check(" in ln:
                    # outside any __global__ body => host wrapper
                    n_host += 1
            bodies[tag] = "\n".join(body)
            gated[tag] = n_host
        if "on" not in bodies or "off" not in bodies:
            return {}
        on, off = bodies["on"], bodies["off"]
        if not on.strip() or not off.strip():
            return {"error": "no __global__ body found"}
        return {
            "device_identical": on == off,
            "device_body_lines": len(on.splitlines()),
            "device_check_calls": {
                "runtime_check": on.count("runtime_check("),
                "choreo_assert": on.count("choreo_assert("),
            },
            "host_gated_checks": gated["on"] - gated["off"],
        }
    except Exception as e:                      # noqa: BLE001 - evidence only
        return {"error": f"{type(e).__name__}: {e}"}


def e5a_one(category, case, shape_class, workdir, reps, device, excl):
    d, err = prep(category, case, workdir)
    base = {
        "toolchain": TOOLCHAIN,
        "category": category,
        "kernel_id": f"{category}/{case}",
        "shape_class": shape_class,
        "settings_hash": run_e2.settings_hash(category),
        "kernel_hash": run_e2.sha1_12(os.path.join(SUITE, category, case + ".co")),
        "toolchain_version": run_e2.toolchain_version(),
        "gpu_device": str(device),
        "exclusive": excl,
        "reps": reps,
    }
    if d is None:
        base["ok"] = False
        base["error"] = err
        return base, []

    # Build BOTH arms before timing anything, so no compile work overlaps a
    # measurement.
    scripts = {}
    for tag, flags in (("on", FLAGS_ON), ("off", FLAGS_OFF)):
        script, err = build(d, flags, tag)
        if script is None:
            base["ok"] = False
            base["error"] = f"build[{tag}]: {err}"
            return base, []
        scripts[tag] = script

    # INTERLEAVED reps, alternating arm order each round.
    #
    # WHY THIS IS NOT OPTIONAL. A first cut ran all `on` reps then all `off`
    # reps and reported checks-on 25% FASTER than checks-off on
    # batch_norm/10_dynamic — physically implausible for six entry guards, and
    # the signature of drift bias: GPU clock ramp, driver/page-cache warm-up and
    # any background load all trend monotonically across a blocked sequence, so
    # whichever arm runs second absorbs the trend. Interleaving makes the two
    # arms see the same drift distribution, and alternating the order each round
    # cancels what remains.
    #
    # The snapshots bracketing the loop are the evidence. If the SM clock moved
    # between them, the Δ below is partly drift and `clock_drift_mhz` says so on
    # the record — a reviewer does not have to take the methodology's word for
    # it. On this machine device 0 idles at 1755 MHz and device 1 at 345 MHz,
    # so the ramp is real, not hypothetical.
    snap_before = gpuinfo.compact_snapshot(device)
    samples = {"on": [], "off": []}
    sources = {"on": set(), "off": set()}
    rcs = {}
    launch = {"on": [], "off": []}
    rejected = {"on": None, "off": None}
    for r in range(reps):
        order = ("on", "off") if r % 2 == 0 else ("off", "on")
        for tag in order:
            if rejected[tag]:
                continue        # deterministic; further reps cannot differ
            rc, so, se, wall = run_exe(scripts[tag],
                                       {"CUDA_VISIBLE_DEVICES": str(device)})
            ok, ev = launch_verdict(rc, so, se)
            launch[tag].append(ev)
            rcs[tag] = rc
            if not ok and ev["reason"] == "launch-rejected":
                # LAUNCH GUARD. This arm's kernel never reached the device, so
                # its millisecond value measures process startup around a no-op.
                # The rejection is a function of the binary and the pinned flags,
                # not of the rep, so remaining reps cannot succeed -- stop now
                # rather than burn exclusive GPU time confirming it.
                rejected[tag] = ev
                continue
            if not ok:
                # Any other non-launch failure keeps the original behaviour: an
                # infra problem on the first rep voids the kernel.
                if len(samples[tag]) == 0:
                    base["ok"] = False
                    base["launch_verdicts"] = {t: launch[t] for t in ("on", "off")}
                    base["error"] = (f"execute[{tag}] rc={rc} "
                                     f"({ev['reason']}): {(se or so)[-200:]}")
                    return base, []
                continue
            ms, src = timing_ms(so, wall)
            samples[tag].append(ms)
            sources[tag].add(src)
    snap_after = gpuinfo.compact_snapshot(device)

    base["gpu_snapshot_before"] = snap_before
    base["gpu_snapshot_after"] = snap_after
    cb, ca = snap_before.get("sm_clock_mhz"), snap_after.get("sm_clock_mhz")
    base["clock_drift_mhz"] = (ca - cb) if (cb is not None and ca is not None) \
        else None

    # ---- LAUNCH GUARD: classify BEFORE any timing is summarised --------------
    # A launch-rejected arm is neither a timed kernel nor an infra failure. It
    # goes into its own bucket, and the pair is emitted with runtime_ms=None so
    # the register carries the EXCLUSION instead of silently dropping the kernel
    # (a dropped kernel is indistinguishable from one that was never selected).
    # See the LAUNCH GUARD block above launch_verdict() for the blast radius
    # this caught: 3 of 14 E5a kernels on the pre-fix binary.
    #
    # Structural evidence for WHERE the gated checks live is computed first so
    # BOTH paths below can carry it. Without it a Δ inside the noise floor is
    # ambiguous ("no cost" vs "cost unresolvable on this host"); with it,
    # `device_identical: true` + `device_check_calls` all zero shows there is no
    # device-side cost to resolve, and the host-side entry guards are ~12
    # integer comparisons against a multi-second launch.
    base["device_code"] = device_code_evidence(scripts)

    if any(rejected.values()):
        arms_rej = sorted(t for t in ("on", "off") if rejected[t])
        first = rejected[arms_rej[0]]
        base["ok"] = False
        base["launch_rejected"] = True
        base["launch_rejected_arms"] = arms_rej
        base["launch_verdicts"] = {t: launch[t] for t in ("on", "off")}
        base["error"] = (f"launch-rejected on arm(s) {','.join(arms_rej)}: "
                         f"{first.get('rejection_text') or first.get('reason')}")
        base["note"] = (
            "The kernel launch was REFUSED by the driver, so no device work "
            "happened and the millisecond value is not a measurement. With "
            "--max-local-mem-capacity=2000000 the dynamic-shape path reserves "
            "capacity * maxThreadsPerSM * numSMs of local memory (466.9 GB on "
            "an H800 PCIe vs 85 GB present), so cudaLaunchKernel returns "
            "cudaErrorInvalidValue. Post-croqtile-1fa4719 this aborts loudly "
            "(rc=134); pre-1fa4719 it exited 0 and still printed an "
            "`Execution time`, which is how three such kernels entered an "
            "earlier E5a as ok=True. Excluded from the residue median, not "
            "averaged into it."
        )
        recs = []
        for checks in ("on", "off"):
            recs.append({
                "toolchain": TOOLCHAIN,
                "category": category,
                "kernel_id": f"{category}/{case}",
                "checks": checks,
                # None, NOT 0.0: 0.0 would be a measurement claiming zero cost.
                "runtime_ms": None,
                "gpu_device": str(device),
                "exclusive": excl,
                "gpu_snapshot": snap_after,
                "clock_drift_mhz": base.get("clock_drift_mhz"),
                "settings_hash": base["settings_hash"],
                "kernel_hash": base["kernel_hash"],
                "toolchain_version": base["toolchain_version"],
                "shape_class": shape_class,
                "timing_source": sorted(sources[checks]) or None,
                "samples_ms": [round(x, 4) for x in samples[checks]],
                "spread_pct": None,
                "reps": reps,
                "launch_ok": False,
                "launch_reason": (rejected[checks] or {}).get("reason")
                                 or "arm-not-reached",
                "launch_evidence": rejected[checks] or launch[checks][-1:],
                "device_code": base["device_code"],
                "definition": E5A_DEFINITION,
            })
        return base, recs

    arms = {}
    for tag in ("on", "off"):
        if not samples[tag]:
            base["ok"] = False
            base["launch_verdicts"] = {t: launch[t] for t in ("on", "off")}
            base["error"] = f"no successful samples for arm {tag}"
            return base, []
        arms[tag] = {
            "runtime_ms": round(statistics.median(samples[tag]), 4),
            "samples_ms": [round(x, 4) for x in samples[tag]],
            "timing_source": sorted(sources[tag]),
            "rc": rcs[tag],
            # Every sample in this arm passed launch_verdict(); say so on the
            # record rather than leaving the reader to infer it from ok=True.
            "launch_ok": True,
            "launch_reps_ok": len(launch[tag]),
        }
        # Spread is reported because a negative or noisy Δ is only interpretable
        # if the reader can see whether it exceeds the arm's own variance.
        arms[tag]["spread_pct"] = round(
            100.0 * (max(samples[tag]) - min(samples[tag]))
            / statistics.median(samples[tag]), 2) if len(samples[tag]) > 1 else 0.0

    on, off = arms["on"]["runtime_ms"], arms["off"]["runtime_ms"]
    base["ok"] = True
    base["checks_on"] = arms["on"]
    base["checks_off"] = arms["off"]
    base["delta_ms"] = round(on - off, 4)
    base["delta_pct"] = round(100.0 * (on - off) / off, 3) if off else None
    base["timing_source"] = arms["on"]["timing_source"]
    # A Δ whose magnitude is inside the arms' own rep-to-rep spread is not a
    # measurement. Flagged so the report can separate signal from noise instead
    # of quoting a sign that a rerun would flip.
    noise = max(arms["on"]["spread_pct"], arms["off"]["spread_pct"])
    base["delta_within_noise"] = bool(
        base["delta_pct"] is not None and abs(base["delta_pct"]) <= noise)
    base["arm_spread_pct"] = noise
    base["launch_ok"] = True

    # Two schema `residue` records — one per arm. The Δ lives on the per-kernel
    # rollup above; the schema's residue record is per (kernel, checks).
    recs = []
    for checks in ("on", "off"):
        recs.append({
            "toolchain": TOOLCHAIN,
            "category": category,
            "kernel_id": f"{category}/{case}",
            "checks": checks,
            "runtime_ms": arms[checks]["runtime_ms"],
            "gpu_device": str(device),
            "exclusive": excl,
            # §12.6 rule 3: the run conditions, not just the number. Clock is
            # recorded per arm so a Δ can be checked against the clock it ran
            # at rather than assumed comparable.
            "gpu_snapshot": snap_after,
            "clock_drift_mhz": base.get("clock_drift_mhz"),
            # --- cross-cutting provenance ---
            "settings_hash": base["settings_hash"],
            "kernel_hash": base["kernel_hash"],
            "toolchain_version": base["toolchain_version"],
            "shape_class": shape_class,
            "timing_source": arms[checks]["timing_source"],
            "samples_ms": arms[checks]["samples_ms"],
            "spread_pct": arms[checks]["spread_pct"],
            "reps": reps,
            # LAUNCH GUARD: this arm's kernel actually reached the device on
            # every rep. Recorded explicitly so stats.py can assert it rather
            # than infer it from the presence of a millisecond value -- the
            # inference is exactly what croqtile 1fa4719 broke.
            "launch_ok": True,
            "launch_reps_ok": arms[checks]["launch_reps_ok"],
            # Structural evidence rides on every residue row so a reader who
            # opens only the schema records (not the per-kernel rollup) still
            # sees WHY a near-zero Δ is expected: the device kernel is identical
            # across arms and carries no checks. Extra fields are permitted by
            # the validator (it checks required + enums only).
            "device_code": base.get("device_code"),
            "definition": E5A_DEFINITION,
        })
    return base, recs


# --------------------------------------------------------------------------
# E5b
# --------------------------------------------------------------------------

def check_loc_of(stdout, stderr):
    """MEASURED entry-vs-interior, per plan §6.2 ("not an assumption").

    Returns (check_loc, report) or (None, None) when the report did NOT come
    from a choreo-emitted check. `None` suppresses the latency record: there is
    no choreo detector to time, and emitting one would credit choreo with a
    detection it did not make.

    THE DISCRIMINATOR IS THE SOURCE-LOCATION SUFFIX. choreo's own generated
    guards append `ar.loc` (`cute_codegen.cpp` EmitPreSiteAssertions /
    EmitPostSiteAssertions both do `ar.message << ", " << ar.loc`), so they
    render as `, <path>:<line>.<col>` — note the DOT before the column. The
    other two `choreo_assert` emitters do not:

      stderr  "choreo runtime check failed: ..., k.co:85.48"
              -> runtime_check(), the generated obligation guard, hoisted to
                 function ENTRY. Measured: all 12 gated sites on
                 layer_normalization/10_dynamic are host-side shape/zero-dim
                 guards at lines 8158-8170, i.e. BEFORE the launch at 8185, and
                 the device kernel body (8077-8155) is byte-identical between
                 the checks-on and checks-off arms. => "entry".

      stdout  "choreo assertion failed: ..., k.co:NN.CC"
              -> choreo_assert() emitted by choreo INSIDE a loop body. Only
                 appears at -rtc>=medium (4 interior guards; default emits 1
                 choreo_assert total and it is the oracle's). => "interior".

      stdout  "choreo assertion failed: <no location>"
              -> NOT a choreo check. Either source B (ArrayProxy::operator[]
                 bounds, runtime/choreo.h:394/405/782/791/884, message
                 "Index out of bounds") or source C (the kernel's own host
                 reference check via the user `assert` BIF, closed vocabulary:
                 "values are not equal.", "concat mismatch", "error", ...).
                 Both are ungated by --disable-runtime-check. => None.

    Verified against the 160 clean `*.run.log` files in raw/e1_logs: 79
    "choreo assertion failed:" lines, and NOT ONE carries a location suffix;
    16 "choreo runtime check failed:" lines, all in the det arm, and the
    gated ones do carry one. So at the pinned default check level the interior
    branch is unreachable — which is the finding, not a gap.
    """
    if RE_RTC.search(stderr or ""):
        return "entry", "choreo runtime check failed"
    if RE_ASSERT.search(stdout or ""):
        msg = ""
        for line in (stdout or "").splitlines():
            if "choreo assertion failed" in line:
                msg = line
                break
        if RE_LOC_SUFFIX.search(msg):
            return "interior", msg.strip()[:160]
        # No location => source B or C, not a choreo check. Returning
        # "interior" here (as an earlier revision did, in BOTH branches, making
        # the regex above dead code) would emit a choreo-entry latency record
        # for a detection choreo did not make.
        return None, msg.strip()[:160] or "choreo assertion failed"
    return None, None


def mutant_source(mut, manifest_paths=None):
    """Resolve an E5b mutant's `.co` source path from an E1 record.

    WHY THIS EXISTS. The first E5b cut did
    `os.path.basename(mut["mutant_path"])` and crashed with
    `KeyError: 'mutant_path'` on the very first mutant — losing the whole run,
    because the artifact is written only at the end of `main()`. E1 records
    (`raw/e1_mutant_records.json`) do NOT carry `mutant_path`; that field lives on
    `raw/mutant_manifest.json` rows. E1 records DO carry `mutant_id`, `class`,
    `category` and `case`, which is enough to rebuild the path: the generator
    writes `mutants/<class>/<category>/<mutant_id>__<case>.co`. Verified 2026-09-09
    against all 16 runtime-outcome mutants: this convention resolves 16/16 and
    agrees with the manifest's `mutant_path` 16/16.

    Order: (1) the reconstructed convention, (2) the manifest path if a manifest
    was loaded, (3) `mut["mutant_path"]` if some future record carries it. Returns
    (path, err); path is None only if nothing resolves.
    """
    cls = mut.get("class")
    cat = mut.get("category")
    mid = mut.get("mutant_id")
    case = mut.get("case")
    tried = []
    if cls and cat and mid and case:
        p = os.path.join(MUTANTS, cls, cat, f"{mid}__{case}.co")
        tried.append(p)
        if os.path.exists(p):
            return p, None
    if manifest_paths and mid in manifest_paths:
        p = manifest_paths[mid]
        if not os.path.isabs(p):
            p = os.path.join(REPO, p)
        tried.append(p)
        if os.path.exists(p):
            return p, None
    if mut.get("mutant_path"):
        p = mut["mutant_path"]
        if not os.path.isabs(p):
            p = os.path.join(REPO, p)
        tried.append(p)
        if os.path.exists(p):
            return p, None
    return None, f"no mutant source resolved; tried {tried}"


def load_manifest_paths():
    """{mutant_id: mutant_path} from raw/mutant_manifest.json, or {} if absent."""
    try:
        with open(MANIFEST) as f:
            m = json.load(f)
        return {r["mutant_id"]: r["mutant_path"]
                for r in m.get("mutants", []) if r.get("mutant_path")}
    except Exception:                                        # noqa: BLE001
        return {}


def e5b_one(mut, workdir, device, excl, manifest_paths=None):
    """One runtime-outcome mutant, both detector arms."""
    mid = mut["mutant_id"]
    cat = mut["category"]
    cls = mut.get("class")
    src, src_err = mutant_source(mut, manifest_paths)
    if src is None:
        rec = {
            "category": cat, "mutant_id": mid, "class": cls,
            "gpu_device": str(device), "exclusive": excl,
            "ok": False, "error": src_err,
        }
        return rec, []
    d, err = prep(cat, mid.replace("/", "_"), workdir, src_override=src)
    rec = {
        "category": cat,
        "mutant_id": mid,
        "class": cls,
        "gpu_device": str(device),
        "exclusive": excl,
        "toolchain_version": run_e2.toolchain_version(),
        "settings_hash": mut.get("settings_hash"),
        "kernel_hash": mut.get("kernel_hash"),
        "mutant_hash": mut.get("mutant_hash"),
    }
    if d is None:
        rec["ok"] = False
        rec["error"] = err
        return rec, []

    # Gate: the mutant must carry a real detector report, else there is no
    # latency to measure. `never` mutants are the ORACLE's business, not this.
    out_recs = []

    # ---- arm (i): choreo's own check, default -rtc -------------------------
    script, err = build(d, FLAGS_ON, "on")
    if script is None:
        rec["ok"] = False
        rec["error"] = f"build[choreo-entry]: {err}"
        return rec, []
    # ---- SYMMETRY FIX (2026-09-09) ---------------------------------------
    # This arm used to time `bash k_on.sh --execute`, which the generated script
    # implements as nvcc-compile THEN run (verified in the script's own
    # dispatch: `--execute` = ${NVCC} ... -o ....exe && ....exe). The sanitizer
    # arm below instead compiles untimed via `compile_then_run` and times only
    # the executable. The two arms therefore measured different regions, and
    # the difference is ~10 s of nvcc: the first full E5b run reported
    #   choreo-entry      median 10.790 s   (min 9.129, max 60.191)
    #   compute-sanitizer median  2.425 s   (min 0.677, max 158.316)
    #   ratio (sanitizer / choreo-entry) = 0.2x
    # i.e. the dynamic oracle looked FASTER than choreo's entry check, exactly
    # inverting the paper's headline E5 claim (L4 §5.6 ¶1: entry checks fail at
    # launch, microseconds-scale and input-independent, while the oracle must
    # run the instrumented body to the fault, up to ~100x). The absolute was
    # never the finding -- the RATIO is -- and a ratio between asymmetric
    # regions is not a measurement at all. Both arms now compile untimed and
    # time only the executable.
    rc, so, se, wall, exe1, err1 = compile_then_run(
        d, script, {"CUDA_VISIBLE_DEVICES": str(device)})
    if err1:
        rec["ok"] = False
        rec["error"] = f"compile-link[choreo-entry]: {err1}"
        return rec, []
    loc, msg = check_loc_of(so, se)
    # LAUNCH GUARD, choreo-entry arm. This arm is EXPECTED to abort: rc=-6 IS
    # the detection (the entry check fires host-side and aborts before the
    # launch), so launch_verdict's rc test must not be read as a failure here.
    # What matters is the opposite case -- if the report is a cudaLaunchKernel
    # rejection rather than a choreo check, then the entry check did NOT catch
    # the fault and the arm timed a rejected launch. check_loc_of already
    # suppresses the record in that case (it returns None unless the report came
    # from a choreo-emitted check), so this records the evidence rather than
    # re-gating.
    ce_launch_ok, ce_launch_ev = launch_verdict(rc, so, se)
    rec["choreo_entry"] = {
        "rc": rc, "wall_s": round(wall, 4), "check_loc": loc,
        "report": msg,
        "detected": bool(RE_RTC.search(se or "")),
        "exe": exe1,
        "timed_region": "executable only (nvcc compile excluded)",
        # `launch_rejected` is the field that matters: True means this arm's
        # abort came from the driver refusing the launch, not from choreo's
        # check. `launch_ok` is recorded for completeness but is EXPECTED False
        # on a detecting arm, so nothing downstream may treat it as a failure.
        "launch_rejected": ce_launch_ev["reason"] == "launch-rejected",
        "launch_evidence": ce_launch_ev,
    }
    if loc:
        out_recs.append({
            "detector": "choreo-entry",
            "category": cat,
            "mutant_id": mid,
            "check_loc": loc,
            "time_to_report_us": round(wall * 1e6, 1),
            "gpu_device": str(device),
            "exclusive": excl,
            # --- cross-cutting provenance ---
            "toolchain": TOOLCHAIN,
            "toolchain_version": rec["toolchain_version"],
            "settings_hash": rec["settings_hash"],
            "kernel_hash": rec["kernel_hash"],
            "class": cls,
            "report": msg,
            # LAUNCH GUARD: carried on the register record so stats.py can
            # quarantine a confounded pair from the register alone, without
            # re-opening the raw artifact.
            "launch_ok": not (ce_launch_ev["reason"] == "launch-rejected"),
            "launch_reason": ce_launch_ev["reason"],
            "definition": E5B_DEFINITION,
        })

    # ---- arm (ii): compute-sanitizer as SOLE detector, -rtc=none -----------
    script2, err = build(d, FLAGS_SANITIZER, "san")
    if script2 is None:
        rec["ok"] = False
        rec["error"] = f"build[sanitizer]: {err}"
        return rec, []
    rc2, so2, se2, wall2, exe, err2 = compile_then_run(
        d, script2, {"CUDA_VISIBLE_DEVICES": str(device)})
    if err2:
        rec["sanitizer"] = {"ok": False, "error": err2}
    else:
        # Re-run under the sanitizer. This is the measured arm.
        t = time.perf_counter()
        try:
            q = subprocess.run(
                ["stdbuf", "-o0", "-e0", SANITIZER,
                 "--tool", "memcheck", "--launch-timeout", "120", exe],
                capture_output=True, timeout=T_EXECUTE, cwd=REPO,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(device)))
            swall = time.perf_counter() - t
            sso = q.stdout.decode("utf8", "replace")
            sse = q.stderr.decode("utf8", "replace")
            src_ = sse + sso
            # "did the tool print a report-prefixed line" -- NOT "did it find a
            # fault". See sanitizer_verdict() for why the distinction matters.
            report_line_seen = bool(RE_SANITIZER.search(src_))
            # The sanitizer arm's OWN exit code. This used to be discarded --
            # `q` went out of scope and only `baseline_rc` (the un-instrumented
            # run) was persisted, so every record showed rc=None. That matters
            # because the detection rule is "exit with no fault report AND
            # rc!=0 -> confounded": without this field the rule cannot be
            # evaluated from the artifact at all, and a reader cannot tell a
            # clean zero-error run (rc=0, body completed, nothing found) from a
            # run that died (rc!=0, no report because it never got there).
            # Those are DIFFERENT claims and only rc separates them.
            src_rc = q.returncode
        except subprocess.TimeoutExpired:
            swall, sso, sse = time.perf_counter() - t, "", "TIMEOUT"
            report_line_seen = False
            src_rc = -1
        first = ""
        for line in (sse + "\n" + sso).splitlines():
            if RE_SANITIZER.search(line):
                first = line.strip()[:200]
                break
        # ---- LAUNCH GUARD + DETECTION GUARD, sanitizer arm ------------------
        # Two independent reasons this arm can produce a wall-clock that is not
        # a time-to-report:
        #   (1) the launch was REJECTED, so the instrumented body never ran and
        #       the wall time measures a refusal (2 of 16, ratios 1.62x/2.32x);
        #   (2) the sanitizer ran the body to completion and reported
        #       "ERROR SUMMARY: 0 errors" -- NO detection at all, so the wall
        #       time measures a full instrumented run that found nothing
        #       (4 of 16, ratios 162x/190x/215x/472x).
        # (2) was invisible because `flagged` was `bool(RE_SANITIZER.search())`
        # and RE_SANITIZER's bare `ERROR` alternative matches the zero-error
        # summary line. See sanitizer_verdict().
        #
        # Note the two act in OPPOSITE directions on the headline ratio: (1)
        # depresses it, (2) inflates it. Quarantining only (1) -- which is what
        # the launch guard alone would do -- moves the median from 64.2x to
        # 143.0x, i.e. it makes the number BETTER by removing the pairs that
        # hurt it while leaving in four pairs where the oracle never reported.
        # Both must go.
        verdict, detected = sanitizer_verdict(first, report_line_seen, rc2,
                                              src_rc)
        bl_launch_ok, bl_launch_ev = launch_verdict(rc2, so2, se2)
        rec["sanitizer"] = {
            # `report_line_seen`, NOT `flagged`. The value is
            # bool(RE_SANITIZER.search(...)) -- "the tool printed a line with a
            # report prefix". Under that name it read as "flagged as faulty",
            # and because RE_SANITIZER's bare `ERROR` alternative also matches
            # the closing "ERROR SUMMARY: 0 errors" line, it was True on runs
            # that found NOTHING. `detected` and `verdict` below carry the
            # interpretation; this field is only the raw observation.
            # (Unrelated to the S12 `sanitizer` record type's `flagged` enum in
            # schema/record-schema.json -- that is a different lane's field and
            # collect.py never reads e5b_detail.)
            "ok": True, "report_line_seen": report_line_seen,
            "wall_s": round(swall, 4),
            "report": first,
            # rc = the SANITIZER arm's exit code; baseline_rc = the
            # un-instrumented run's. Both are kept because the detection rule
            # needs to distinguish them: a zero-error report with rc=0 means the
            # body ran to completion and nothing was found (oracle blind), while
            # rc!=0 with no report means the arm died before reporting.
            "rc": src_rc,
            "baseline_rc": rc2,
            "baseline_wall_s": round(wall2, 4),
            "exe": exe,
            # Same timed region as the choreo-entry arm (see SYMMETRY FIX).
            "timed_region": "executable only (nvcc compile excluded)",
            # The guard's own output. `verdict` names WHY, `detected` says
            # whether there is anything to time.
            "verdict": verdict,
            "detected": detected,
            "baseline_launch_ok": bl_launch_ok,
            "baseline_launch_evidence": bl_launch_ev,
        }
        # A latency record is emitted for EVERY arm that ran, detecting or not,
        # but non-detections carry `quarantine_reason` so stats.py can hold them
        # out of the ratio while still reporting the bucket. Dropping them
        # instead would make the exclusion invisible -- and an invisible
        # exclusion is indistinguishable from a pair that was never measured.
        out_recs.append({
            "detector": "compute-sanitizer",
            "category": cat,
            "mutant_id": mid,
            # The sanitizer instruments every access and reports when the
            # faulting access EXECUTES — interior by construction. Recorded
            # as measured because the schema demands a value, and because
            # the contrast with choreo's hoisted entry check IS the finding.
            "check_loc": "interior",
            "time_to_report_us": round(swall * 1e6, 1),
            "gpu_device": str(device),
            "exclusive": excl,
            "toolchain": TOOLCHAIN,
            "toolchain_version": rec["toolchain_version"],
            "settings_hash": rec["settings_hash"],
            "kernel_hash": rec["kernel_hash"],
            "class": cls,
            "report": first,
            # --- LAUNCH / DETECTION GUARD ---
            # `rc` is carried on the REGISTER record, not only on e5b_detail,
            # because stats.py's guard must be able to re-derive the verdict from
            # the register alone. The register is what collect.py writes and what
            # a reader audits; if the discriminator lived only in the raw
            # artifact the guard would be unverifiable from the published data.
            "rc": src_rc,
            "detected": detected,
            "sanitizer_verdict": verdict,
            "launch_ok": bl_launch_ok and verdict != "launch-rejected",
            "launch_reason": ("launch-rejected" if verdict == "launch-rejected"
                              else bl_launch_ev["reason"]),
            "quarantine_reason": (None if detected and verdict != "launch-rejected"
                                  else verdict),
            "definition": E5B_DEFINITION,
        })

    rec["ok"] = True
    return rec, out_recs


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=1,
                    help="MUST stay 1. This is a timing lane: concurrent runs "
                         "contend for the device and invalidate every sample.")
    ap.add_argument("--reps", type=int, default=5,
                    help="E5a execute reps per arm; median reported")
    ap.add_argument("--size", default="small", choices=["small", "full"],
                    help="small = one dynamic case per category; full = every "
                         "dynamic-shape kernel in the suite")
    ap.add_argument("--include-static", action="store_true",
                    help="also measure static-shape cases. Plan §6 asserts their "
                         "residue is 0 by construction; E2/E3 found they DO emit "
                         "entry shape guards, so this tests the premise instead "
                         "of assuming it.")
    ap.add_argument("--only", default="", help="restrict E5a to one category")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    ap.add_argument("--skip-e5b", action="store_true")
    ap.add_argument("--skip-e5a", action="store_true",
                    help="reuse the banked E5a checkpoint instead of re-measuring "
                         "it, and run only E5b. E5a is the expensive half (~27 min "
                         "of GPU-exclusive sampling that needs the machine to "
                         "itself); E5b is cheap. When an E5b-only bug is found and "
                         "fixed, re-measuring E5a would burn the exclusive window "
                         "for nothing and would also REPLACE good E5a data with a "
                         "fresh noisy draw. Reads --e5a-checkpoint.")
    ap.add_argument("--e5a-checkpoint", default=None,
                    help="E5a checkpoint to reuse under --skip-e5a. Default: the "
                         "`<out>.e5a.json` sibling of --out, which is exactly "
                         "where the checkpoint block writes it.")
    ap.add_argument("--e1", default=E1_RECORDS)
    ap.add_argument("--workdir", default=WORKDIR)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--keep-logs", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="measure even though the host is contended; every "
                         "record is stamped `host_contended` so the integrator "
                         "cannot mistake it for a clean sample")
    ap.add_argument("--wait-quiet", type=int, default=1800,
                    help="seconds to wait for competing nvcc/cicc/choreo "
                         "processes to finish before giving up. 0 = check once "
                         "and abort immediately (the old behaviour). Default "
                         "1800 because a transient burst from another user's "
                         "job should not kill a ~2 h unattended run.")
    a = ap.parse_args()

    a.workdir = os.path.abspath(a.workdir)
    a.out = os.path.abspath(a.out)
    a.e1 = os.path.abspath(a.e1)
    os.makedirs(a.workdir, exist_ok=True)

    if a.jobs != 1:
        print("WARNING: --jobs > 1 on a timing lane contaminates every sample. "
              "Proceeding, but the numbers are not publishable.", file=sys.stderr)

    if not os.path.exists(CHOREO):
        print(f"ERROR: no choreo binary at {CHOREO}", file=sys.stderr)
        return 2

    excl = lock_held()
    if not excl:
        print("*** WARNING: benchmark2/.gpu-lock is NOT held. ***")
        print("    E5 is a timing lane; records are stamped exclusive=false and")
        print("    qc WILL REJECT them. Run via `run.sh e5`, which takes the")
        print("    lock, or create it manually before measuring.")

    ver = run_e2.toolchain_version()
    print(f"[choreo] E5: runtime cost & latency (device={a.device}, "
          f"exclusive={excl}, reps={a.reps}, size={a.size})")

    # ============================================================== E5a ====
    # The checkpoint path is computed up front because --skip-e5a READS it and
    # the checkpoint block below WRITES it; they must agree.
    ckpt = os.path.join(os.path.dirname(a.out) or ".",
                        os.path.basename(a.out).replace(".json", "") + ".e5a.json")
    e5a_kernels, residue_records, e5a_bad = [], [], []
    e5a_launch_rejected = []
    e5a_s = 0.0
    e5a_reused_from = None
    e5a_toolchain_mismatch = None
    if a.skip_e5a:
        src_ck = os.path.abspath(a.e5a_checkpoint) if a.e5a_checkpoint else ckpt
        e5a_reused_from = src_ck
        if not os.path.exists(src_ck):
            print(f"ERROR: --skip-e5a given but there is no E5a checkpoint at "
                  f"{src_ck}. Run E5a once (without --skip-e5a) to bank it.",
                  file=sys.stderr)
            return 2
        with open(src_ck) as f:
            ck = json.load(f)
        e5a_kernels = ck.get("e5a_kernels", [])
        residue_records = ck.get("residue_records", [])
        e5a_bad = ck.get("e5a_failures", [])
        e5a_launch_rejected = ck.get("e5a_launch_rejected", [])
        e5a_s = (ck.get("elapsed_s") or {}).get("e5a", 0.0)
        # Inherit the banked run's sampling depth so the artifact's `reps` field
        # describes the data it actually carries, not this invocation's flag.
        a.reps = ck.get("reps", a.reps)
        print(f"[choreo] E5a: SKIPPED (--skip-e5a) — reusing {len(e5a_kernels)} "
              f"banked kernel(s) from {os.path.relpath(src_ck, REPO)} "
              f"(exclusive={ck.get('exclusive')}, reps={a.reps}, "
              f"{e5a_s}s of sampling, {len(e5a_bad)} failure(s))")
        if ck.get("host_contended"):
            print(f"    NOTE: the banked E5a was measured while the host was "
                  f"contended ({','.join(ck['host_contended'])}); its records "
                  f"carry that stamp.")
        if ck.get("partial") != "e5a-only" and ck.get("latency_records"):
            print("    NOTE: the source looks like a COMPLETE artifact, not an "
                  "E5a checkpoint. Its E5a half is still what gets reused.")
        # ---- MIXED-TOOLCHAIN GUARD (added 2026-09-09, after it bit us) ----
        # --skip-e5a can silently produce an artifact whose two halves were
        # measured by DIFFERENT binaries. It happened for real: E5a was banked
        # at 14:09 from a binary built from 680533f, the binary was rebuilt at
        # 14:57, and the E5b-only rerun at ~15:45 stamped f2f238f. The merged
        # artifact's top-level `toolchain_version` then said f2f238f while all
        # 28 residue records said 680533f -- and nothing in the run complained.
        # Per-record stamps stayed honest, but a reader of the top-level field
        # would have been misled, and E5a's residue numbers are not comparable
        # to E5b's latency numbers if the codegen changed between them.
        # WARN LOUDLY. Do not abort: reusing E5a across a rebuild is sometimes
        # exactly what you want (E5a is 27 min of exclusive GPU time), and the
        # per-record stamps preserve the truth. But the operator must decide,
        # not have it decided for them silently.
        ck_ver = ck.get("toolchain_version")
        if ck_ver and ver and ck_ver != ver:
            print(f"    *** MIXED TOOLCHAIN: the banked E5a was measured by "
                  f"binary {ck_ver[:12]}, but THIS run's binary is "
                  f"{ver[:12]}. ***", file=sys.stderr)
            print(f"        The merged artifact will carry {ver[:12]} at the "
                  f"top level while its {len(residue_records)} residue records "
                  f"carry {ck_ver[:12]}.", file=sys.stderr)
            print(f"        E5a (residue) and E5b (latency) are then NOT from "
                  f"the same toolchain. Do not quote them as one measurement "
                  f"without saying so.", file=sys.stderr)
            print(f"        To get a single-toolchain artifact, re-run E5a too "
                  f"(drop --skip-e5a) after the rebuild settles.",
                  file=sys.stderr)
            e5a_toolchain_mismatch = {"banked": ck_ver, "this_run": ver}
        else:
            e5a_toolchain_mismatch = None
        ks = []
    else:
        ks = dynamic_kernels(include_static=a.include_static)
        if ks is None:
            print(f"ERROR: {E2_LEDGER} missing — run E2 first "
                  f"(it supplies the authoritative shape_class).",
                  file=sys.stderr)
            return 2
        if a.only:
            ks = [k for k in ks if k[0] == a.only]
        if a.size == "small":
            seen = {}
            picked = []
            for cat, case, sc in ks:
                if seen.setdefault((cat, sc), None) is None:
                    seen[(cat, sc)] = case
                    picked.append((cat, case, sc))
            ks = picked
        if a.limit:
            ks = ks[:a.limit]
        print(f"[choreo] E5a: {len(ks)} kernel(s) x 2 arms x {a.reps} reps "
              f"(interleaved, arm order alternated)")

    # The contention gate applies to BOTH halves: E5b is a latency measurement
    # too, so it must not run on a contended host even when E5a is skipped.
    busy = host_busy()
    if busy and a.wait_quiet > 0 and not a.force:
        print(f"[choreo] host contended ({','.join(busy)}); waiting up to "
              f"{a.wait_quiet}s for it to clear ...", flush=True)
        busy, waited = wait_for_quiet(a.wait_quiet)
        if not busy:
            print(f"[choreo] host quiet after {waited:.0f}s", flush=True)
    if busy:
        print(f"*** HOST CONTENDED: competing process(es) alive: {busy} ***")
        print("    Every sample below is contaminated. Wait for them to finish")
        print("    and re-run; do not publish these numbers.")
        if not a.force:
            print("    (pass --force to measure anyway and stamp "
                  "`host_contended` on every record, or --wait-quiet N to "
                  "block until the host clears)")
            return 3
    if not a.skip_e5a:
        # NOTE: do NOT re-initialise e5a_kernels / residue_records / e5a_bad
        # here. Under --skip-e5a they already hold the banked checkpoint data,
        # and clearing them would silently discard the E5a half this run exists
        # to preserve. They are initialised once, above, before the skip branch.
        t0 = time.time()
        for i, (cat, case, sc) in enumerate(ks, 1):
            rec, recs = e5a_one(cat, case, sc, a.workdir, a.reps, a.device,
                                excl)
            if rec.get("ok"):
                e5a_kernels.append(rec)
                residue_records.extend(recs)
                print(f"  [{i:>3}/{len(ks)}] {rec['kernel_id'][:52]:<54} "
                      f"on={rec['checks_on']['runtime_ms']:9.3f}ms "
                      f"off={rec['checks_off']['runtime_ms']:9.3f}ms "
                      f"delta={rec['delta_pct']:+7.3f}%  "
                      f"({','.join(rec['timing_source'])})")
            elif rec.get("launch_rejected"):
                # LAUNCH GUARD: its own bucket, NOT e5a_bad. A rejected launch
                # is the auditor working as designed, not an infra failure, and
                # lumping the two would let a reader dismiss the exclusion as a
                # flaky run. The residue records still go into the register --
                # with runtime_ms=None -- so the exclusion is visible there.
                e5a_launch_rejected.append(rec)
                residue_records.extend(recs)
                print(f"  [{i:>3}/{len(ks)}] {cat}/{case:<50} "
                      f"LAUNCH-REJECTED ({','.join(rec['launch_rejected_arms'])})"
                      f" — excluded from residue, not timed")
            else:
                e5a_bad.append(rec)
                print(f"  [{i:>3}/{len(ks)}] {cat}/{case} FAILED: "
                      f"{rec.get('error','')[:100]}")
            if not a.keep_logs:
                subprocess.run(["rm", "-rf", os.path.join(a.workdir,
                                                          f"{cat}_{case}")])
        e5a_s = round(time.time() - t0, 1)
        if e5a_launch_rejected:
            print(f"\n*** LAUNCH GUARD: {len(e5a_launch_rejected)} of {len(ks)} "
                  f"E5a kernel(s) had their launch REFUSED by the driver "
                  f"(cudaErrorInvalidValue on cudaLaunchKernel). ***")
            print("    These are NOT timed kernels and are NOT in "
                  "`e5a_kernels`. Their residue rows carry runtime_ms=null so "
                  "the register shows the exclusion rather than dropping it.")
            print("    Cause: --max-local-mem-capacity=2000000 makes the "
                  "dynamic-shape path reserve capacity*maxThreadsPerSM*numSMs "
                  "(466.9 GB) against 85 GB of device memory.")
            for r in e5a_launch_rejected:
                print(f"      - {r['kernel_id']}  "
                      f"arms={','.join(r['launch_rejected_arms'])}")

    # ---- CHECKPOINT: bank E5a before E5b runs ------------------------------
    # WHY. The artifact used to be written only at the very end of main(). When
    # E5b raised KeyError on its first mutant, the exception propagated out and
    # NO artifact was created -- ~40 min of GPU-exclusive E5a sampling (15
    # kernels x 2 arms x 5 reps, needing the machine to itself) was destroyed by
    # a bug in a different stage. E5a is the expensive, lock-holding half; E5b is
    # cheap. So E5a is flushed to disk here, and the final write below overwrites
    # it with the complete artifact. A reader must be able to tell the two apart:
    # the checkpoint carries `partial: "e5a-only"` and an empty E5b section.
    #
    # Skipped under --skip-e5a: this run measured no E5a data, so writing the
    # checkpoint would overwrite the very file it was loaded from. The banked
    # E5a must survive an E5b-only re-run untouched.
    if not a.skip_e5a:
        try:
            with open(ckpt, "w") as f:
                json.dump({
                    "toolchain": TOOLCHAIN, "toolchain_version": ver,
                    "produced_by": "e5", "partial": "e5a-only",
                    "size": a.size, "gpu_device": str(a.device),
                    "exclusive": excl, "reps": a.reps,
                    "elapsed_s": {"e5a": e5a_s},
                    "host_contended": busy,
                    "residue_records": residue_records,
                    "e5a_kernels": sorted(e5a_kernels,
                                          key=lambda r: r["kernel_id"]),
                    "e5a_failures": e5a_bad,
                    "e5a_launch_rejected": sorted(
                        e5a_launch_rejected, key=lambda r: r["kernel_id"]),
                }, f, indent=1)
            print(f"[choreo] checkpointed E5a ({len(e5a_kernels)} kernels) to "
                  f"{os.path.relpath(ckpt, REPO)}")
        except Exception as e:                               # noqa: BLE001
            # A failed checkpoint must not abort the run; warn and carry on.
            print(f"[choreo] *** checkpoint write FAILED "
                  f"({type(e).__name__}: {e}); continuing without an E5a backup")

    # ============================================================== E5b ====
    e5b_detail, latency_records, e5b_note = [], [], None
    e5b_s = 0.0
    if a.skip_e5b:
        e5b_note = "skipped by --skip-e5b"
    elif not os.path.exists(a.e1):
        e5b_note = (f"SKIPPED: no E1 records at {os.path.relpath(a.e1, REPO)}. "
                    f"E5b measures latency on E1's runtime-outcome mutants, so "
                    f"E1 must run first (plan work cycle: level-1 E1 is the "
                    f"decisive first stage).")
        print(f"[choreo] E5b: {e5b_note}")
    else:
        with open(a.e1) as f:
            e1 = json.load(f)
        rows = e1.get("records", e1 if isinstance(e1, list) else [])
        # Plan §6.2: "on E1's `r`-outcome (dynamic-shape) mutants". In the schema
        # enum that is outcome=runtime. `never` mutants are the oracle's business
        # and have no detector latency to measure.
        cand = [r for r in rows if r.get("outcome") == "runtime"]
        print(f"[choreo] E5b: {len(cand)} runtime-outcome mutant(s) from E1 "
              f"({len(rows)} records total)")
        if not cand:
            e5b_note = ("no E1 mutant had outcome=runtime, so there is no "
                        "detector latency to measure. This is itself a result: "
                        "choreo discharged every mutant statically (see S2's "
                        "`never = 0` assertion) — report it, do not fabricate "
                        "a latency row.")
            print(f"[choreo] E5b: {e5b_note}")
        t1 = time.time()
        manifest_paths = load_manifest_paths()
        for i, mut in enumerate(cand, 1):
            # Per-mutant isolation. WHY: the first E5b cut raised KeyError on
            # mutant #1 and the exception propagated out of main(), so the
            # artifact -- which is written only at the END of main() -- was never
            # created. ~40 min of GPU-exclusive E5a sampling was lost to a bug in
            # a completely different stage. One bad mutant must cost one mutant.
            try:
                rec, recs = e5b_one(mut, a.workdir, a.device, excl,
                                    manifest_paths)
            except Exception as e:                           # noqa: BLE001
                rec = {
                    "category": mut.get("category"),
                    "mutant_id": mut.get("mutant_id"),
                    "class": mut.get("class"),
                    "gpu_device": str(a.device),
                    "exclusive": excl,
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                }
                recs = []
                print(f"  *** E5b crashed on {rec['mutant_id']}: {rec['error']}")
            e5b_detail.append(rec)
            latency_records.extend(recs)
            ce = rec.get("choreo_entry", {})
            sn = rec.get("sanitizer", {})
            # `san=` shows the VERDICT, not `flagged`. `flagged` is
            # bool(RE_SANITIZER.search(...)) and RE_SANITIZER's bare `ERROR`
            # alternative matches compute-sanitizer's closing
            # "ERROR SUMMARY: 0 errors" line -- so `flagged` is True on a run
            # that found NOTHING. Printing it here made the console claim a
            # detection on 4 of 16 arms, which is exactly the reading that
            # produced the void 472x headline. The operator watches this line
            # for ~27 min; it must say what the guard concluded.
            print(f"  [{i:>3}/{len(cand)}] {rec['mutant_id'][:46]:<48} "
                  f"choreo={ce.get('check_loc','-'):<9}"
                  f"{'Y' if ce.get('detected') else 'n'} "
                  f"san={'Y' if sn.get('detected') else 'n'} "
                  f"{sn.get('verdict','')}")
            if not a.keep_logs:
                subprocess.run(["rm", "-rf",
                                os.path.join(a.workdir,
                                             f"{rec['category']}_"
                                             f"{rec['mutant_id'].replace('/','_')}")])
        e5b_s = round(time.time() - t1, 1)

    # ------------------------------------------------------------- output ----
    dyn = [r for r in e5a_kernels if r["shape_class"] == "dynamic"]
    stat = [r for r in e5a_kernels if r["shape_class"] == "static"]

    def med(rs, key):
        v = [r[key] for r in rs if r.get(key) is not None]
        return round(statistics.median(v), 4) if v else None

    out = {
        "toolchain": TOOLCHAIN,
        "toolchain_version": ver,
        # E1, E2 and E4 all carry this; E5 did not, and that omission is why the
        # mixed-binary state below went unnoticed. `binary_sha1_12` is the only
        # field that pins WHICH BUILD ran, as opposed to which commit the mtime
        # heuristic guessed at. Without it, two artifacts claiming the same
        # `toolchain_version` can have been produced by different binaries and
        # nothing in the record shows it.
        "toolchain_identity": toolchain.identity(),
        "produced_by": "e5",
        "size": a.size,
        "gpu_device": str(a.device),
        "exclusive": excl,
        "reps": a.reps,
        "elapsed_s": {"e5a": e5a_s, "e5b": e5b_s},
        "e5a_definition": E5A_DEFINITION,
        "e5b_definition": E5B_DEFINITION,
        "e5b_note": e5b_note,
        # PROVENANCE: under --skip-e5a the E5a half was not measured by this
        # invocation. Say so, and say where it came from, so a reader cannot
        # mistake a reused E5a for a fresh draw. The per-residue-record
        # `exclusive` stamps are the banked ones (True), which is what
        # collect.py validates and stats.py uses for admissibility; the
        # top-level `exclusive`/`host_contended` below describe THIS
        # invocation, i.e. the E5b half.
        "e5a_reused": (None if not e5a_reused_from else {
            "from": os.path.relpath(e5a_reused_from, REPO),
            "banked_exclusive": ck.get("exclusive") if a.skip_e5a else None,
            "banked_host_contended": (ck.get("host_contended")
                                      if a.skip_e5a else None),
            "banked_elapsed_s": e5a_s,
            "banked_toolchain_version": (ck.get("toolchain_version")
                                         if a.skip_e5a else None),
            "toolchain_mismatch": e5a_toolchain_mismatch,
            "note": "E5a loaded from the banked checkpoint; only E5b was "
                    "measured by this invocation."
                    + ("" if not e5a_toolchain_mismatch else
                       " *** THE TWO HALVES ARE FROM DIFFERENT BINARIES -- see "
                       "toolchain_mismatch. The top-level toolchain_version "
                       "describes E5b only; the residue records describe E5a. "
                       "Do not quote them as one measurement without saying "
                       "so."),
        }),
        "host_contended": busy,
        "summary": {
            "dynamic_kernels": len(dyn),
            "static_kernels": len(stat),
            "median_delta_pct_dynamic": med(dyn, "delta_pct"),
            "median_delta_ms_dynamic": med(dyn, "delta_ms"),
            "median_delta_pct_static": med(stat, "delta_pct"),
            "latency_pairs": len([r for r in latency_records
                                  if r["detector"] == "choreo-entry"]),
            # LAUNCH GUARD roll-up. Kept at the top of the artifact so a reader
            # sees the exclusions before any median.
            "launch_guard": {
                "e5a_launch_rejected": len(e5a_launch_rejected),
                "e5a_launch_rejected_kernels": sorted(
                    r["kernel_id"] for r in e5a_launch_rejected),
                "e5b_sanitizer_quarantined": sum(
                    1 for r in latency_records
                    if r.get("detector") == "compute-sanitizer"
                    and r.get("quarantine_reason")),
                "e5b_quarantine_breakdown": dict(collections.Counter(
                    r.get("quarantine_reason") for r in latency_records
                    if r.get("detector") == "compute-sanitizer"
                    and r.get("quarantine_reason"))),
                "definition": (
                    "launch-rejected = the driver refused cudaLaunchKernel, so "
                    "no device work happened and the wall time is not a "
                    "measurement. no-fault-reported = compute-sanitizer ran the "
                    "body to completion and printed 'ERROR SUMMARY: 0 errors', "
                    "so there is no time-to-report. Both are EXCLUDED from the "
                    "E5b ratio and from the E5a residue median; neither is "
                    "averaged in, and neither is silently dropped."),
            },
        },
        "residue_records": residue_records,
        "latency_records": latency_records,
        "e5a_kernels": sorted(e5a_kernels, key=lambda r: r["kernel_id"]),
        "e5a_failures": e5a_bad,
        "e5a_launch_rejected": sorted(e5a_launch_rejected,
                                      key=lambda r: r["kernel_id"]),
        "e5b_detail": e5b_detail,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {os.path.relpath(a.out, REPO)}")

    # ------------------------------------------------------------- report ----
    print("\n=== S13 / E5a: residue overhead (checks on vs off) ===")
    print(f"{'shape class':<12}{'n':>4}{'median Δ%':>12}{'median Δms':>13}"
          f"{'min Δ%':>10}{'max Δ%':>10}{'in-noise':>10}")
    for label, rs in (("dynamic", dyn), ("static", stat)):
        if not rs:
            print(f"{label:<12}{0:>4}{'—':>12}{'—':>13}{'—':>10}{'—':>10}{'—':>10}")
            continue
        dp = [r["delta_pct"] for r in rs if r["delta_pct"] is not None]
        nz = sum(1 for r in rs if r.get("delta_within_noise"))
        print(f"{label:<12}{len(rs):>4}{statistics.median(dp):>12.3f}"
              f"{med(rs,'delta_ms'):>13.4f}{min(dp):>10.3f}{max(dp):>10.3f}"
              f"{nz:>7}/{len(rs)}")

    noisy = [r for r in e5a_kernels if r.get("delta_within_noise")]
    if noisy:
        print(f"\n*** {len(noisy)}/{len(e5a_kernels)} Δ are INSIDE the arms' own "
              f"rep-to-rep spread ***")
        print("    Their sign is not a measurement. Do not quote a per-case Δ")
        print("    from this set; quote the median across cases and say so.")
        for r in noisy[:8]:
            print(f"      {r['kernel_id'][:56]:<58} Δ={r['delta_pct']:+7.3f}% "
                  f"spread={r['arm_spread_pct']:.1f}%")

    neg = [r for r in e5a_kernels if (r["delta_pct"] or 0) < 0]
    if neg:
        print(f"\n*** {len(neg)} case(s) show checks-on FASTER than checks-off ***")
        print("    Six entry guards cannot make a kernel faster. Either the Δ is")
        print("    inside the noise floor (see above) or the host was contended.")
        print("    This lane refuses to present such a Δ as guard overhead.")

    if stat:
        print("\n*** PLAN §6.2 PREMISE TEST: 'residue is 0 by construction on "
              "static-shape cases' ***")
        nz = [r for r in stat if (r["delta_pct"] or 0) > 0.5]
        print(f"  {len(stat)} static case(s) measured; {len(nz)} show Δ > 0.5%.")
        print("  E2/E3 already found static kernels emit 3-8 entry shape guards")
        print("  that -rtc=none removes, so a nonzero Δ here is EXPECTED and")
        print("  must be reported to the coordinator: it qualifies §6.2 and the")
        print("  E5a scoping claim, and it is not this lane's to absorb.")

    if latency_records:
        print("\n=== S13 / E5b: detection latency ===")
        by = {}
        for r in latency_records:
            by.setdefault(r["detector"], []).append(r["time_to_report_us"])
        for det in ("choreo-entry", "compute-sanitizer"):
            v = by.get(det, [])
            if v:
                print(f"  {det:<20} n={len(v):<4} median={statistics.median(v):12.1f} us"
                      f"  min={min(v):10.1f}  max={max(v):12.1f}")
        if len(by) == 2:
            a_ = statistics.median(by["choreo-entry"])
            b_ = statistics.median(by["compute-sanitizer"])
            print(f"  ratio (sanitizer / choreo-entry): {b_/a_:.1f}x")
        locs = {}
        for r in latency_records:
            if r["detector"] == "choreo-entry":
                locs[r["check_loc"]] = locs.get(r["check_loc"], 0) + 1
        print(f"  choreo check_loc (MEASURED, not assumed): {locs}")
    elif e5b_note:
        print(f"\n=== S13 / E5b: {e5b_note}")

    if e5a_bad:
        print(f"\n*** {len(e5a_bad)} E5a kernel(s) FAILED (excluded, not zero) ***")
        for r in e5a_bad[:10]:
            print(f"    {r['kernel_id']}: {r.get('error','')[:120]}")

    if not excl:
        print("\n*** exclusive=false: these records will be REJECTED by qc. ***")
    return 1 if e5a_bad else 0


if __name__ == "__main__":
    sys.exit(main())
