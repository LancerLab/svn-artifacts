#!/usr/bin/env python3
"""Calibrate the specs §7 ground-truth oracle for every base case E1 uses.

specs §7 requires a manifest check: run unmutated vs mutated on the reference
input and confirm the output differs. Every choreo kernel already carries an
independent host-side reference in its own `main()` (or in the category's
`common.h`/`common.hpp` behind `#ifdef __CHECK__`), so no separate reference
harness is needed. But that reference is only usable if it PASSES ON CORRECT
CODE. This script establishes exactly that, per (category, case).

Measured facts that make this step mandatory, not optional:

  * softmax/1_bert_32x512x768_32x512x768 — the UNMUTATED base aborts with
    "choreo assertion failed: Test Failed" under -D__CHECK__. Its cpu_softmax1
    reference disagrees with the kernel's own layout. Enabling __CHECK__ there
    would mark every mutant `corrupts` no matter what the defect is.
  * matmul/1_bert_... — the UNMUTATED base fails to COMPILE at the default
    2048-byte local-memory cap (exit=4) and passes cleanly at the pinned
    2000000 cap, printing "Test matmul1 Passed!". So the cap is required even
    for correct code, and __CHECK__ is usable.

Outputs raw/oracle_policy.json:

  {"<category>/<case>": {
      "base_compile":   bool,   # unmutated compiles at the pinned cap
      "base_run_rc":    int,
      "base_passed":    bool,   # unmutated prints a success marker, checks off
      "check_passed":   bool,   # same, with -D__CHECK__ (gated categories)
      "use_check":      bool,   # __CHECK__ is enabled AND the base passes with it
      "oracle_usable":  bool,   # we can distinguish corrupts from noop at all
      "reason":         str}}

`oracle_usable` is false when the base itself fails: then a mutant's failure is
not evidence of corruption, and run_e1.py must report manifest verdicts for that
case as untrustworthy rather than emitting a number.

Usage: calibrate_oracle.py [--jobs N] [--only CATEGORY]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_e1 as E1  # noqa: E402  (reuse flags, regexes, run())

OUT = os.path.join(E1.RAW, "oracle_policy.json")
SUITE = os.path.join(E1.REPO, "benchmark", "choreo")


def base_cases_used():
    """The (category, case) pairs the current mutant set actually touches."""
    data = json.load(open(E1.MANIFEST_IN))
    seen = set()
    for r in data["mutants"]:
        seen.add((r["category"], r["case"]))
    return sorted(seen)


def survey_cases(categories):
    """Every case in the given categories, whether or not a mutant uses it.

    Used to re-point mutations.BASE_CASES at a base whose oracle is usable.
    Mutating a kernel that already fails its own reference check gives no clean
    signal: the mutant's failure is not evidence of the injected defect.
    """
    out = []
    for cat in categories:
        d = os.path.join(SUITE, cat)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith(".co"):
                out.append((cat, name[:-3]))
    return out


def calibrate_one(category, case, workdir):
    key = f"{category}/{case}"
    d = os.path.join(workdir, key.replace("/", "_"))
    os.makedirs(d, exist_ok=True)

    # Copy the base kernel and its category headers into a scratch dir so the
    # generated .cu resolves `#include "common.hpp"` against the right place.
    src = os.path.join(SUITE, category, case + ".co")
    local = os.path.join(d, case + ".co")
    shutil.copy2(src, local)
    for h in E1.local_headers_for(category, SUITE):
        shutil.copy2(h, d)

    res = {"category": category, "case": case, "gated": category in E1.GATED_CHECK}

    sh = os.path.join(d, "base.sh")
    ok, rc, ctext = E1.compile_stage(local, E1.ORACLE_FLAGS, sh,
                                     os.path.join(d, "base.compile.log"))
    res["base_compile"] = ok
    res["base_compile_rc"] = rc
    if not ok:
        res.update(base_run_rc=None, base_passed=False, check_passed=False,
                   use_check=False, oracle_usable=False,
                   reason=f"unmutated base does not compile (rc={rc})")
        return key, res

    # checks off, no __CHECK__: does correct code reach a success marker?
    rc0, t0, infra0 = E1.execute_stage(sh, os.path.join(d, "base.run.log"))
    caught0, passed0 = E1.classify_log(t0, rc0)
    res["base_run_rc"] = rc0
    res["base_passed"] = passed0
    res["base_detector"] = caught0
    res["base_infra"] = infra0

    passed_check = None
    if res["gated"]:
        rc1, t1, infra1 = E1.execute_stage(
            sh, os.path.join(d, "base.check.run.log"),
            env={"EXTRA_TARGET_CFLAGS": "-D__CHECK__"})
        caught1, passed1 = E1.classify_log(t1, rc1)
        res["check_run_rc"] = rc1
        res["check_passed"] = passed1
        res["check_detector"] = caught1
        passed_check = passed1
    else:
        res["check_run_rc"] = rc0
        res["check_passed"] = passed0
        res["check_detector"] = caught0
        passed_check = passed0

    res["use_check"] = bool(res["gated"] and passed_check)
    # The oracle is usable iff correct code passes under the configuration we
    # will actually use for mutants.
    effective_pass = passed_check if res["gated"] else passed0
    res["oracle_usable"] = bool(effective_pass)
    if res["oracle_usable"]:
        res["reason"] = ("__CHECK__ reference passes on unmutated code"
                         if res["use_check"] else
                         "built-in reference passes on unmutated code")
    else:
        res["reason"] = ("UNMUTATED base fails its own reference check — "
                         "manifest verdicts for this case are untrustworthy")
    return key, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument("--workdir", default=os.path.join(E1.RAW, "oracle_calibration"))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--survey", default="",
                    help="comma-separated categories: calibrate EVERY case in "
                         "them, not just the ones the mutant set uses")
    a = ap.parse_args()

    if a.survey:
        cats = [c.strip() for c in a.survey.split(",") if c.strip()]
        pairs = survey_cases(cats)
    else:
        pairs = base_cases_used()
        if a.only:
            pairs = [(c, k) for (c, k) in pairs if c == a.only]
    if a.survey and a.only:
        pairs = [(c, k) for (c, k) in pairs if c == a.only]
    # MUST be absolute: E1.run() executes choreo with cwd=REPO, so a relative
    # scratch path resolves against the repo root and choreo reports
    # "The input file ... does not exist" (rc=1) for every case.
    a.workdir = os.path.abspath(a.workdir)
    a.out = os.path.abspath(a.out)
    os.makedirs(a.workdir, exist_ok=True)
    print(f"[choreo] calibrating oracle on {len(pairs)} unmutated base cases "
          f"with {a.jobs} workers")

    policy = {}
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(calibrate_one, c, k, a.workdir) for c, k in pairs]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            key, res = fut.result()
            policy[key] = res
            flag = "OK " if res["oracle_usable"] else "BAD"
            print(f"  [{i:>2}/{len(pairs)}] {flag} {key:<58} "
                  f"compile={str(res['base_compile']):<5} "
                  f"base_passed={str(res['base_passed']):<5} "
                  f"check_passed={str(res['check_passed']):<5} "
                  f"use_check={res['use_check']}", flush=True)

    with open(a.out, "w") as f:
        json.dump({"toolchain": E1.TOOLCHAIN,
                   "toolchain_version": E1.toolchain_version(),
                   "produced_by": "calibrate_oracle",
                   "policy": {k: policy[k] for k in sorted(policy)}}, f, indent=1)
    print(f"\nwrote {os.path.relpath(a.out, E1.REPO)}")

    bad = [k for k, v in sorted(policy.items()) if not v["oracle_usable"]]
    print(f"\noracle usable: {len(policy) - len(bad)}/{len(policy)}")
    if bad:
        print("UNUSABLE (base fails its own reference — no manifest verdict possible):")
        for k in bad:
            print(f"  {k:<58} {policy[k]['reason']}")
    gated_ok = [k for k, v in sorted(policy.items()) if v["use_check"]]
    print(f"\n__CHECK__ enabled for: {len(gated_ok)} cases")
    for k in gated_ok:
        print(f"  {k}")

    if a.survey:
        print("\n=== per-category usable bases (candidates for BASE_CASES) ===")
        bycat = {}
        for k, v in policy.items():
            bycat.setdefault(v["category"], []).append((v["case"], v))
        for cat in sorted(bycat):
            good = [(c, v) for c, v in sorted(bycat[cat]) if v["oracle_usable"]]
            print(f"\n{cat}: {len(good)}/{len(bycat[cat])} usable")
            for c, v in good:
                tag = " (needs -D__CHECK__)" if v["use_check"] else ""
                print(f"    {c}{tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
