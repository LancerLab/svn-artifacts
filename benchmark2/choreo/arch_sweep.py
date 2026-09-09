#!/usr/bin/env python3
"""Is choreo's obligation analysis arch-dependent?

Motivation. The host has 2x H800 (compute capability 9.0) but every generated
build script hardcodes `nv_arch=sm_86`, because the lane never passes `-arch`.
choreo does accept `-arch=<processor|native>`, and `-arch=native` auto-detects
sm_90a. So the mismatch is fixable by a flag rather than a rebuild -- but only
if fixing it does not invalidate measurements already taken.

Two lanes are exposed differently:

  * E2/E3 (obligation ledger, S3-S7) is STATIC ANALYSIS. If the arch choice
    changes which obligations exist, their outcomes, or the numeric hardware
    limits they assert, then the whole 17,353-record corpus was measured under
    the wrong architecture and must be redone. If it changes only the message
    label ("On SM_86" vs "On SM_90A"), the corpus stands and the defect is
    presentational.

  * E4/E5 (timing) is CODEGEN. `-arch=sm_86` on an sm_90 device means no SASS
    is produced for the actual GPU, so the driver JIT-compiles PTX at load.
    That contaminates any latency figure regardless of what the ledger says.

This script settles the first question by brute force: dump the ledger for every
kernel under both architectures and compare them field by field, after
normalizing the arch label. Anything that differs beyond the label is reported
per kernel, because that is a real analysis difference and not a string artifact.

Deliberately narrow in what it treats as "the same". Obligations are compared as
multisets of (function, loc, outcome, usage, dependence, mechanism, cost,
enabled) plus the message with the arch label stripped, so a reordering in the
ledger does not masquerade as a difference and a genuine limit change cannot
hide inside a label normalization.

Usage:
    python3 arch_sweep.py [--jobs N] [--kernels-glob PATTERN] [--out FILE]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CHOREO = os.path.join(REPO, "croqtile", "build-release", "choreo")
BENCH = os.path.join(REPO, "benchmark", "choreo")

CAP = "--max-local-mem-capacity=2000000"
# Match run_e2.py's LEDGER_FLAGS: static analysis only, no nvcc, no GPU.
LEDGER_FLAGS = ["-gs", "--stats", "-es", CAP, "-t", "cute"]

# "On SM_86," / "On SM_90A," -- the only place the arch name reaches the ledger.
RE_ARCH_LABEL = re.compile(r"\bSM_\d+[A-Z]?\b")

# Fields that carry analysis meaning. `message` is handled separately because it
# embeds the arch label.
KEY_FIELDS = ("function", "loc", "outcome", "usage", "dependence", "mechanism",
              "cost", "enabled")


def dump_ledger(src: str, arch: str | None) -> tuple[list | None, str]:
    """Return (obligations, error). arch=None means choreo's default."""
    flags = list(LEDGER_FLAGS)
    if arch:
        flags.append(f"-arch={arch}")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        led = tf.name
    out_sh = led + ".sh"
    cmd = [CHOREO] + flags + ["--dump-ledger=" + led, src, "-o", out_sh]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                           cwd=REPO, errors="replace")
    except subprocess.TimeoutExpired:
        return None, "timeout"
    finally:
        for f in (out_sh,):
            if os.path.exists(f):
                os.unlink(f)
    if not os.path.exists(led):
        err = (p.stderr or p.stdout or "").strip().splitlines()
        return None, (err[-1][:200] if err else f"rc={p.returncode}, no ledger")
    try:
        with open(led) as fh:
            d = json.load(fh)
    except Exception as ex:                                    # noqa: BLE001
        return None, f"unparseable: {ex}"
    finally:
        if os.path.exists(led):
            os.unlink(led)
    obs = d if isinstance(d, list) else d.get("obligations", d.get("ledger", []))
    return obs, ""


def fingerprint(obs: list) -> tuple:
    """Arch-insensitive, order-insensitive fingerprint of an obligation set."""
    rows = []
    for o in obs:
        msg = RE_ARCH_LABEL.sub("SM_XX", str(o.get("message", "")))
        rows.append(tuple([str(o.get(k)) for k in KEY_FIELDS] + [msg]))
    return tuple(sorted(rows))


def numeric_limits(obs: list) -> Counter:
    """Every integer >= 1000 appearing in a message: the hardware limits."""
    c: Counter = Counter()
    for o in obs:
        for n in re.findall(r"\d{4,}", str(o.get("message", ""))):
            c[n] += 1
    return c


def arch_labels(obs: list) -> set:
    out = set()
    for o in obs:
        out.update(RE_ARCH_LABEL.findall(str(o.get("message", ""))))
    return out


def one(src: str) -> dict:
    rel = os.path.relpath(src, REPO)
    a, ea = dump_ledger(src, None)
    b, eb = dump_ledger(src, "native")
    rec: dict = {"kernel": rel, "error": None}
    if a is None or b is None:
        rec["error"] = f"default={ea or 'ok'} native={eb or 'ok'}"
        return rec
    rec["n_default"] = len(a)
    rec["n_native"] = len(b)
    rec["labels_default"] = sorted(arch_labels(a))
    rec["labels_native"] = sorted(arch_labels(b))
    rec["same_after_label_norm"] = fingerprint(a) == fingerprint(b)
    la, lb = numeric_limits(a), numeric_limits(b)
    rec["limits_default"] = dict(la)
    rec["limits_native"] = dict(lb)
    rec["limits_same"] = la == lb
    if not rec["same_after_label_norm"]:
        # Show what actually moved, so the report is actionable rather than a
        # bare "differs".
        fa, fb = fingerprint(a), fingerprint(b)
        only_a = [r for r in fa if r not in set(fb)]
        only_b = [r for r in fb if r not in set(fa)]
        rec["only_default"] = [list(r) for r in only_a[:6]]
        rec["only_native"] = [list(r) for r in only_b[:6]]
        rec["n_only_default"] = len(only_a)
        rec["n_only_native"] = len(only_b)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--kernels-glob", default="*/*.co",
                    help="glob under benchmark/choreo (default: all kernels)")
    ap.add_argument("--out", default=os.path.join(HERE, "raw", "arch_sweep.json"))
    a = ap.parse_args()

    if not os.path.exists(CHOREO):
        print(f"ERROR: no choreo binary at {CHOREO}", file=sys.stderr)
        return 2

    import glob as globmod
    srcs = sorted(globmod.glob(os.path.join(BENCH, a.kernels_glob)))
    if not srcs:
        print(f"ERROR: no kernels matched {a.kernels_glob} under {BENCH}",
              file=sys.stderr)
        return 2
    print(f"[arch] {len(srcs)} kernels x 2 arches, jobs={a.jobs} "
          f"(static analysis only -- no nvcc, no GPU)")

    recs: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(one, s): s for s in srcs}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            recs.append(fut.result())
            if i % 25 == 0 or i == len(srcs):
                print(f"[arch] {i}/{len(srcs)}", flush=True)

    recs.sort(key=lambda r: r["kernel"])
    errs = [r for r in recs if r["error"]]
    ok = [r for r in recs if not r["error"]]
    label_only = [r for r in ok if r["same_after_label_norm"] and r["limits_same"]]
    real_diff = [r for r in ok if not (r["same_after_label_norm"] and r["limits_same"])]

    payload = {
        "produced_by": "arch_sweep.py",
        "question": "does -arch change choreo's obligation ANALYSIS, or only "
                    "the arch label in the message?",
        "choreo": CHOREO,
        "flags_default": LEDGER_FLAGS,
        "flags_native": LEDGER_FLAGS + ["-arch=native"],
        "n_kernels": len(recs),
        "n_errors": len(errs),
        "n_label_only": len(label_only),
        "n_real_analysis_difference": len(real_diff),
        "verdict": ("ARCH-INDEPENDENT ANALYSIS: every difference is the message "
                    "label alone, so the E2/E3 obligation corpus is valid as "
                    "measured and only the quoted label is wrong"
                    if not real_diff and not errs else
                    "ARCH-DEPENDENT: see real_analysis_differences"),
        "labels_seen_default": sorted({l for r in ok for l in r["labels_default"]}),
        "labels_seen_native": sorted({l for r in ok for l in r["labels_native"]}),
        "real_analysis_differences": real_diff,
        "errors": errs,
        "records": recs,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=1)

    print(f"\n[arch] kernels={len(recs)} errors={len(errs)} "
          f"label-only={len(label_only)} REAL-DIFF={len(real_diff)}")
    print(f"[arch] labels: default={payload['labels_seen_default']} "
          f"native={payload['labels_seen_native']}")
    print(f"[arch] verdict: {payload['verdict']}")
    for r in real_diff[:10]:
        print(f"  DIFF {r['kernel']}: n {r['n_default']}->{r['n_native']} "
              f"limits_same={r['limits_same']} "
              f"only_default={r.get('n_only_default')} only_native={r.get('n_only_native')}")
    for r in errs[:5]:
        print(f"  ERR  {r['kernel']}: {r['error']}")
    print(f"[arch] wrote {os.path.relpath(a.out, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
