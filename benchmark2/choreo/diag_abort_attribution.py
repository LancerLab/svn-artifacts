#!/usr/bin/env python3
"""Who actually killed the E5b "process-abort" arms -- memcheck, or choreo?

WHY THIS EXISTS.
E5b's premise (run_e5.py:191) is that the sanitizer arm is built with
--disable-runtime-check "so the sanitizer is the sole detector". Six of the 16
arms came back with report == "========= Error: process didn't terminate
successfully" and were classified `process-abort, detected=True` -- i.e. counted
as an ORACLE detection and timed as an oracle time-to-report.

But `baseline_rc` for all six is -6. The baseline arm runs the SAME executable
(FLAGS_SANITIZER == FLAGS_OFF) WITHOUT compute-sanitizer. If it aborts there
too, the abort cannot have been caused by the sanitizer. And E1's oracle-arm
logs name the killer directly:

    M1.s2.rl1.read          choreo.h:221  "values are not equal"
    M1.s2.rl11.read         choreo.h:221  "values are not equal"
    M1.s2.tp1.read          choreo.h:221  "error"
    M1.s2.tp11.read         choreo.h:221  "error"
    M2.s1.ln1.caller.scale  choreo.h:886  "Index out of bounds"
    M2.s1.ln3.caller.scale  choreo.h:886  "Index out of bounds"

choreo.h:221 is `choreo_assert`; :886 is `ArrayProxy::operator[]`'s bounds
check. Neither is gated by --disable-runtime-check (they are the four-source
model's sources B and C -- ungated by construction). So on those six arms the
process was killed by CHOREO's residual ungated assertion, not by the dynamic
oracle, and the recorded wall time measures the wrong instrument.

RESOLVED (2026-09-09). This probe confirmed the confound, and the classifier was
then fixed at the source: sanitizer_verdict returns detected=False for
process-abort (run_e5.py EDIT 5), stats.py's launch_status quarantines it, and
detection_asymmetry gives it a fifth bucket plus a premise_violation_note. The
truly admissible E5b latency set is now n=4 (memcheck-fault only), median
84.45x. The `detected=True` framing in the paragraphs above describes the state
this probe was written to investigate, not the current behaviour. E5b's premise
that --disable-runtime-check leaves "the sanitizer as sole detector" is FALSE
and is reported, not designed around (directive item 1).

WHAT THIS SCRIPT DOES. For each named mutant it rebuilds with the exact E5b
sanitizer-arm flags and captures the FULL output of three runs:
  1. plain exe, no sanitizer          -> does it abort on its own? (baseline)
  2. exe under compute-sanitizer      -> does memcheck report a fault BEFORE
                                         the abort, or only the abort line?
  3. grep of the sanitizer output for any memcheck fault line at all

Run 2 is the decisive one. `first` in run_e5.py is the FIRST line matching
RE_SANITIZER, so an abort line there means no fault line preceded it -- but
memcheck reports at synchronisation points, so an abort could in principle
pre-empt a report it had already recorded. Only the full text settles it.

This is a correctness probe, not a timing lane: it does not hold the GPU lock
and its wall-clock numbers mean nothing. It compiles, so it checks host_busy()
BEFORE spawning nvcc (gpuinfo's documented limitation: `ps -eo comm=` cannot
distinguish our own children from someone else's).
"""

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)

import gpuinfo          # noqa: E402
import run_e5           # noqa: E402

# The six arms classified `process-abort` in raw/e5_runtime.json.
TARGETS = [
    "M1.s2.rl1.read",
    "M1.s2.rl11.read",
    "M1.s2.tp1.read",
    "M1.s2.tp11.read",
    "M2.s1.ln1.caller.scale",
    "M2.s1.ln3.caller.scale",
]

# A memcheck fault line, as opposed to the tool's own bookkeeping. This is
# deliberately WIDER than run_e5.RE_SANITIZER_FAULT so that a fault memcheck
# reported in some other phrasing is not missed by the probe.
RE_ANY_MEMCHECK_FAULT = re.compile(
    r"Invalid __global__|Invalid __shared__|Invalid __local__"
    r"|out-of-bounds|misaligned|unspecified launch failure"
    r"|Racecheck|Barrier error|Leaked", re.I)

# choreo's own assertion voices -- sources A-device, B and C.
RE_CHOREO_ASSERT = re.compile(r"choreo assertion failed", re.I)
RE_CHOREO_RTC = re.compile(r"choreo runtime check failed", re.I)

OUT = "/tmp/diag_abort_attribution.json"


def main():
    busy = gpuinfo.host_busy()
    if busy:
        # A warning, not an abort. gpuinfo.host_busy() exists for TIMING lanes,
        # where a concurrent nvcc saturating every core makes a wall-clock
        # meaningless (that is how a first E5a cut came to report checks-on 25%
        # FASTER than checks-off). This probe reads exit codes and report text,
        # not wall clocks -- contention can slow it but cannot change a verdict.
        # It still checks BEFORE spawning any nvcc of its own, per gpuinfo's
        # documented limitation that `ps -eo comm=` cannot tell our children
        # from someone else's; the point here is only to say so out loud.
        print(f"NOTE: host busy ({', '.join(busy)}). Wall-clock numbers below "
              f"are therefore meaningless -- but this probe reads exit codes "
              f"and report text, which contention cannot change. Proceeding.")

    e1 = json.load(open(run_e5.E1_RECORDS))
    recs = {r["mutant_id"]: r for r in e1["records"]}
    paths = run_e5.load_manifest_paths()

    work = "/tmp/diag_abort_work"
    os.makedirs(work, exist_ok=True)
    results = []

    for mid in TARGETS:
        mut = recs.get(mid)
        if not mut:
            print(f"  {mid}: NOT in E1 records -- skipped")
            continue
        src, err = run_e5.mutant_source(mut, paths)
        if err:
            print(f"  {mid}: {err} -- skipped")
            continue

        d, perr = run_e5.prep(mut["category"], mid, work, src_override=src)
        if perr:
            print(f"  {mid}: prep failed: {perr}")
            continue
        script, berr = run_e5.build(d, run_e5.FLAGS_SANITIZER, "san")
        if berr:
            print(f"  {mid}: build failed: {berr[:200]}")
            continue
        exe = run_e5.exe_path(script)
        if not exe:
            print(f"  {mid}: no exe path in script")
            continue

        env = {"CUDA_VISIBLE_DEVICES": "0"}

        # 1. plain exe -- no sanitizer. Does it abort on its own?
        #    compile_then_run, NOT run_exe: run_e5.py:1097 builds baseline_rc
        #    with compile_then_run (--compile-link then run the .exe directly).
        #    run_exe would go through `--execute`, which recompiles with nvcc
        #    first -- a different code path and a different timed region. The
        #    probe must reproduce the arm that produced the number.
        rc1, so1, se1, _w1, exe, cerr = run_e5.compile_then_run(
            d, script, env)
        if cerr:
            print(f"  {mid}: compile-link failed: {cerr[:200]}")
            continue

        # 2. under compute-sanitizer -- the decisive run.
        try:
            q = subprocess.run(
                ["stdbuf", "-o0", "-e0", run_e5.SANITIZER,
                 "--tool", "memcheck", "--launch-timeout", "120", exe],
                capture_output=True, timeout=run_e5.T_EXECUTE,
                cwd=run_e5.REPO, env=dict(os.environ, **env))
            rc2 = q.returncode
            full2 = (q.stderr.decode("utf8", "replace") + "\n"
                     + q.stdout.decode("utf8", "replace"))
        except subprocess.TimeoutExpired:
            rc2, full2 = -1, "TIMEOUT"

        san_lines = [l.strip() for l in full2.splitlines()
                     if l.startswith("=========")]
        has_memcheck_fault = bool(RE_ANY_MEMCHECK_FAULT.search(full2))
        has_choreo_assert = bool(RE_CHOREO_ASSERT.search(full2))
        has_choreo_rtc = bool(RE_CHOREO_RTC.search(full2))

        # Reproduce run_e5.py's `first` selection EXACTLY -- the first line
        # matching RE_SANITIZER, not the first "=========" line. Those differ:
        # compute-sanitizer's banner ("========= COMPUTE-SANITIZER") does not
        # match RE_SANITIZER, so feeding san_lines[0] to the classifier yields
        # `unclassified-but-flagged` while the instrument itself would have
        # yielded `process-abort`. A probe that misreports the verdict it is
        # auditing is worse than no probe.
        first = ""
        for line in full2.splitlines():
            if run_e5.RE_SANITIZER.search(line):
                first = line.strip()[:200]
                break

        # Likewise derive `report_line_seen` instead of hardcoding True. The
        # instrument computes bool(RE_SANITIZER.search(stderr + stdout)); a
        # probe that assumes True would report a verdict the instrument could
        # never have produced on an arm where the tool printed nothing matching.
        report_line_seen = bool(run_e5.RE_SANITIZER.search(full2))

        # The plain run's own voice. THIS is the decisive evidence: if choreo's
        # assertion fires with NO sanitizer present, the abort is choreo's, and
        # the sanitizer arm's wall time measured choreo, not the oracle.
        plain_blob = (se1 or "") + "\n" + (so1 or "")
        plain_choreo_assert = bool(RE_CHOREO_ASSERT.search(plain_blob))
        plain_choreo_rtc = bool(RE_CHOREO_RTC.search(plain_blob))
        # Name the killer line, e.g. "choreo.h:886: choreo assertion failed:
        # Index out of bounds". The line number discriminates source A-device
        # (gated) from sources B and C (ungated) in the four-source model.
        m = re.search(r"choreo\.h:(\d+): choreo assertion failed: ([^\n]*)",
                      plain_blob)
        plain_killer = (f"choreo.h:{m.group(1)}: {m.group(2).strip()[:60]}"
                        if m else None)

        # Mirror run_e5.py:1155 exactly: sanitizer_verdict(first,
        # report_line_seen, baseline_rc, arm_rc). rc1 is the plain (baseline)
        # run, rc2 the sanitizer arm -- the same order the instrument uses.
        verdict, detected = run_e5.sanitizer_verdict(
            first, report_line_seen, rc1, rc2)

        row = {
            "mutant_id": mid,
            "category": mut["category"],
            "plain_exe_rc": rc1,
            "plain_exe_aborts_alone": rc1 != 0,
            "plain_choreo_assertion": plain_choreo_assert,
            "plain_choreo_runtime_check": plain_choreo_rtc,
            "plain_killer_line": plain_killer,
            "sanitizer_rc": rc2,
            "n_sanitizer_lines": len(san_lines),
            "sanitizer_lines": san_lines[:12],
            # What was actually fed to the classifier. Recording these makes
            # the verdict auditable from the artifact alone: a reader can see
            # the exact `first` line and `report_line_seen` the instrument
            # would have seen, rather than having to trust the verdict.
            "classifier_first_line": first,
            "classifier_report_line_seen": report_line_seen,
            "memcheck_fault_line_present": has_memcheck_fault,
            "choreo_assertion_in_output": has_choreo_assert,
            "choreo_runtime_check_in_output": has_choreo_rtc,
            "current_verdict": verdict,
            "current_detected": detected,
        }
        results.append(row)

        print(f"\n  {mid}")
        print(f"    plain exe rc={rc1}  (aborts WITHOUT sanitizer: "
              f"{rc1 != 0})")
        if plain_killer:
            print(f"    plain-run killer: {plain_killer}")
        print(f"    sanitizer rc={rc2}  memcheck-fault-line={has_memcheck_fault}"
              f"  choreo-assert-in-output={has_choreo_assert}")
        print(f"    current verdict={verdict} detected={detected}")
        for l in san_lines[:6]:
            print(f"      | {l[:110]}")

    n_abort_alone = sum(1 for r in results if r["plain_exe_aborts_alone"])
    n_no_memcheck = sum(1 for r in results
                        if not r["memcheck_fault_line_present"])
    n_choreo = sum(1 for r in results if r["plain_choreo_assertion"])

    print("\n" + "=" * 70)
    print(f"  {len(results)} mutant(s) probed")
    print(f"  abort with NO sanitizer present : {n_abort_alone}")
    print(f"  NO memcheck fault line at all   : {n_no_memcheck}")
    print(f"  choreo assertion in PLAIN run   : {n_choreo}")
    killers = sorted({r["plain_killer_line"] for r in results
                      if r["plain_killer_line"]})
    if killers:
        print("  killer line(s), plain run:")
        for k in killers:
            print(f"    - {k}")
    if n_abort_alone == len(results) and n_no_memcheck == len(results):
        print("\n  VERDICT: CONFOUNDED. On every probed arm the executable "
              "aborts on its own and memcheck reports no fault. The abort is "
              "choreo's own UNGATED assertion (source B/C), not the dynamic "
              "oracle. sanitizer_verdict now returns detected=False for "
              "process-abort (run_e5.py EDIT 5) and stats.py quarantines it, so "
              "these arms no longer credit the oracle with choreo's detection "
              "and no longer enter the latency ratio.")
    elif n_no_memcheck == 0:
        print("\n  VERDICT: CLEAN. memcheck reported a real fault on every "
              "probed arm; process-abort is a legitimate oracle detection.")
    else:
        print("\n  VERDICT: MIXED. Per-mutant rows above decide; do not pool.")

    with open(OUT, "w") as f:
        json.dump({"targets": TARGETS, "results": results,
                   "n_abort_alone": n_abort_alone,
                   "n_no_memcheck_fault": n_no_memcheck,
                   "n_choreo_assertion_plain": n_choreo,
                   "plain_killer_lines": killers}, f, indent=2)
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
