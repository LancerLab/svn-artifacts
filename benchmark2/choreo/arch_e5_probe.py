#!/usr/bin/env python3
"""Does `-arch` change what E5a MEASURES? (diagnostic, not a lane)

WHY THIS EXISTS
---------------
choreo's codegen defaults to `nv_arch=sm_86` while this host is sm_90 (H800,
compute_cap 9.0). `arch_sweep.py` already settled the STATIC question: the
obligation analysis is arch-independent (309/311 kernels differ only in the
`SM_86`/`SM_90A` label inside a message). E4 is immune too — it times
`--compile-link`, i.e. nvcc's CPU work, and never executes.

E5a is the one lane that EXECUTES, so it is the one lane where arch could
matter. This probe answers that empirically instead of by argument.

TWO SEPARATE WORRIES, AND WHY THEY ARE NOT THE SAME
----------------------------------------------------
1. **JIT cost.** `-arch=sm_86` on sm_90 means nvcc emits sm_86 SASS. cuobjdump
   confirms the binary carries `sm_86.cubin` x2 AND `sm_86.ptx` x1, so the
   driver CAN and DOES JIT the PTX at module load. But module load is process
   startup, and E5a's primary timing source is the kernel's own
   `Execution time: N us` marker, which brackets the choreo call and EXCLUDES
   startup. So JIT should not enter the marker at all. Measured here.

2. **SASS quality.** JIT'd sm_86->sm_90 code may allocate registers and select
   instructions differently from native sm_90a SASS, which could change how
   expensive the guards are. This DOES land inside the marker. Measured here.

WHAT WOULD MAKE ARCH IRRELEVANT TO E5a
--------------------------------------
E5a reports a DELTA: checks-on minus checks-off, both arms compiled at the SAME
arch. Arch is therefore common-mode. If the delta is stable across arches, then
E5a's headline number is arch-independent even though its absolute times are
not — and the lane can run as-is, with the caveat recorded.

If the delta MOVES with arch, E5a must be re-run at `-arch=native` and any
previously banked residue numbers are void.

METHOD
------
For each kernel, build all FOUR combinations (2 arches x 2 arms) up front, then
execute them in a ROTATING order so that every combination occupies every
position in the sequence equally often. This is the same defense `run_e5.py`
uses (interleaved reps, alternating arm order each round) and it is not
optional here either.

WHY ROTATION IS REQUIRED. A first cut of this probe blocked by arm — all `on`
reps, then all `off` reps, within each arch. It reported delta 19 ms at
sm_86 vs 173 ms at sm_90a and concluded "arch moves the delta". That conclusion
was an artifact: this host idles at 345 MHz with a 1755 MHz max and
persistence_mode Disabled, so clock ramp, driver warm-up and page-cache effects
all trend monotonically across a blocked sequence, and whichever combination
runs later absorbs the trend. `run_e5.py` documents the identical failure mode
(a first cut reported checks-on 25% FASTER than checks-off, physically
implausible for six entry guards). Rotation makes all four combinations see the
same drift distribution.

Reports, per kernel:
  * absolute median per (arch, arm), the timing_source used, and each
    combination's own rep-to-rep `spread_pct`
  * delta_on_off per arch, and whether the two deltas agree
  * `delta_within_noise`: whether each delta is smaller than its own arms'
    spread. A delta inside the noise floor is NOT a measurement, so a
    "disagreement" between two such deltas proves nothing about arch.
  * process wall-clock too, so the JIT/startup component is visible separately
    from the marker — if arch moves wall-clock but not the marker, that is
    exactly the JIT-is-startup prediction confirming itself
  * `clock_drift_mhz` bracketing the whole loop, so a reviewer can see whether
    the GPU changed clock state during the measurement

This is a DIAGNOSTIC. It writes no schema records and feeds no statistic. Run it
under the GPU lock (`benchmark2/.gpu-lock`) or its numbers mean nothing.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

import gpuinfo

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
B2 = os.path.join(REPO, "benchmark2")
CHOREO = os.path.join(REPO, "croqtile", "build-release", "choreo")
SUITE = os.path.join(REPO, "benchmark", "choreo")
LOCK = os.path.join(B2, ".gpu-lock")
DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

CAP = "--max-local-mem-capacity=2000000"
BASE_FLAGS = ["-gs", "-t", "cute", "-kt", CAP]
FLAGS_ON = BASE_FLAGS
FLAGS_OFF = BASE_FLAGS + ["--disable-runtime-check"]

T_COMPILE = 1800
T_EXECUTE = 900

# Same regex E5a uses, so this probe measures what the lane measures.
RE_EXECTIME = re.compile(r"Execution time:\s*(\d+)\s*(?:us|microseconds)", re.I)
RE_ARCH = re.compile(r"nv_arch=(sm_[0-9a]+)")


def prep(category, case, tag):
    """Copy kernel + local headers into a scratch dir (E5's own trap: the
    generated CUDA resolves `#include "common.h"` via -I<dir of the .co>)."""
    d = os.path.join("/tmp", f"archprobe_{category}_{case}_{tag}")
    os.makedirs(d, exist_ok=True)
    src = os.path.join(SUITE, category, case + ".co")
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


def build(d, flags, arch, arm):
    out = os.path.join(d, f"k_{arm}_{arch}.sh")
    cmd = [CHOREO] + flags + ([f"-arch={arch}"] if arch != "default" else []) \
        + [os.path.join(d, "k.co"), "-o", out]
    r = subprocess.run(cmd, capture_output=True, timeout=T_COMPILE, cwd=REPO)
    if r.returncode != 0 or not os.path.exists(out):
        return None, None, f"choreo rc={r.returncode}: " \
            + r.stderr.decode("utf8", "replace")[-300:]
    with open(out, "rb") as f:
        m = RE_ARCH.search(f.read().decode("utf8", "replace"))
    return out, (m.group(1) if m else "?"), None


def run(script):
    """`stdbuf -o0 -e0` is MANDATORY: abort() discards buffered stdout.

    DEVICE is pinned in the env exactly as `run_e5.py` does, so this probe and
    the lane measure the same GPU. Both devices idle at 345 MHz with
    persistence_mode Disabled, so an unpinned run could land on either and the
    two arches would not be comparable.
    """
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(DEVICE))
    t = time.perf_counter()
    try:
        p = subprocess.run(["stdbuf", "-o0", "-e0", "bash", script, "--execute"],
                           capture_output=True, timeout=T_EXECUTE, cwd=REPO,
                           env=env)
        wall = time.perf_counter() - t
        return p.returncode, p.stdout.decode("utf8", "replace"), wall
    except subprocess.TimeoutExpired:
        return -1, "", time.perf_counter() - t


def marker_ms(stdout):
    vals = [int(m.group(1)) for m in RE_EXECTIME.finditer(stdout)]
    if not vals:
        return None
    return statistics.median(vals) / 1000.0


def one_kernel(category, case, reps, rounds):
    """Rotating A/B across arches. Returns a per-kernel record.

    All four (arch, arm) combinations are built up front, then executed in a
    rotating order so each occupies each sequence position equally often. See
    the module docstring for why blocking by arm produces a fake result on this
    host.
    """
    rec = {"category": category, "case": case, "reps": reps, "rounds": rounds,
           "arms": {}}
    d, err = prep(category, case, "work")
    if err:
        rec["error"] = err
        return rec

    built = {}
    for arch in ("default", "native"):
        for arm, flags in (("on", FLAGS_ON), ("off", FLAGS_OFF)):
            script, nv_arch, berr = build(d, flags, arch, arm)
            if berr:
                rec["error"] = f"build {arch}/{arm}: {berr}"
                return rec
            built[(arch, arm)] = (script, nv_arch)
    rec["nv_arch"] = {f"{a}/{arm}": v[1] for (a, arm), v in built.items()}

    # The four combinations, in a FIXED base order. Each round rotates that
    # order by one position, so over `rounds` rounds every combination has
    # occupied every slot. With 4 combinations, 4 rounds gives a complete
    # Latin-square-style balance; fewer rounds still beats blocking.
    combos = [("default", "on"), ("default", "off"),
              ("native", "on"), ("native", "off")]

    samples = {k: [] for k in combos}
    walls = {k: [] for k in combos}
    positions = {k: [] for k in combos}   # sequence slot each sample occupied

    snap_before = gpuinfo.compact_snapshot(DEVICE)
    slot = 0
    for rd in range(rounds):
        order = combos[rd % len(combos):] + combos[:rd % len(combos)]
        for k in order:
            script, _ = built[k]
            for _ in range(reps):
                rc, so, wall = run(script)
                slot += 1
                ms = marker_ms(so)
                if ms is None:
                    rec.setdefault("no_marker", []).append(f"{k[0]}/{k[1]}")
                    continue
                samples[k].append(ms)
                walls[k].append(wall)
                positions[k].append(slot)
    snap_after = gpuinfo.compact_snapshot(DEVICE)

    cb = snap_before.get("sm_clock_mhz")
    ca = snap_after.get("sm_clock_mhz")
    rec["clock_drift_mhz"] = (ca - cb) if (cb is not None and ca is not None) else None
    rec["gpu_snapshot_before"] = snap_before
    rec["gpu_snapshot_after"] = snap_after

    for k, (script, nv_arch) in built.items():
        arch, arm = k
        s, w = samples[k], walls[k]
        med = statistics.median(s) if s else None
        spread = (round(100.0 * (max(s) - min(s)) / med, 2)
                  if (s and med and len(s) > 1) else 0.0)
        rec["arms"][f"{arch}/{arm}"] = {
            "nv_arch": nv_arch,
            "n": len(s),
            "marker_ms_median": (round(med, 4) if med is not None else None),
            "marker_ms_min": (round(min(s), 4) if s else None),
            "marker_ms_max": (round(max(s), 4) if s else None),
            "spread_pct": spread,
            "wall_s_median": (round(statistics.median(w), 4) if w else None),
            # mean sequence slot: if this is far from the others, rotation
            # failed and drift bias is back in play.
            "mean_position": (round(statistics.mean(positions[k]), 2)
                              if positions[k] else None),
        }

    def delta(arch, field):
        a = rec["arms"].get(f"{arch}/on", {}).get(field)
        b = rec["arms"].get(f"{arch}/off", {}).get(field)
        if a is None or b is None:
            return None
        return round(a - b, 4)

    rec["delta_marker_ms"] = {"default": delta("default", "marker_ms_median"),
                              "native": delta("native", "marker_ms_median")}
    rec["delta_wall_s"] = {"default": delta("default", "wall_s_median"),
                           "native": delta("native", "wall_s_median")}

    # A delta smaller than its own arms' rep-to-rep spread is NOT a measurement.
    # Two such deltas "disagreeing" says nothing about arch, so this is recorded
    # per arch and gates the verdict.
    for arch in ("default", "native"):
        on = rec["arms"].get(f"{arch}/on", {})
        off = rec["arms"].get(f"{arch}/off", {})
        dms = rec["delta_marker_ms"][arch]
        noise = max(on.get("spread_pct", 0.0), off.get("spread_pct", 0.0))
        dpct = (round(100.0 * dms / off["marker_ms_median"], 3)
                if (dms is not None and off.get("marker_ms_median")) else None)
        rec.setdefault("delta_within_noise", {})[arch] = {
            "delta_ms": dms,
            "delta_pct": dpct,
            "arm_spread_pct": noise,
            "within_noise": bool(dpct is not None and abs(dpct) <= noise),
        }

    d_def, d_nat = rec["delta_marker_ms"]["default"], rec["delta_marker_ms"]["native"]
    wn = rec["delta_within_noise"]
    both_noisy = wn["default"]["within_noise"] and wn["native"]["within_noise"]
    if d_def is not None and d_nat is not None:
        rec["delta_abs_diff_ms"] = round(abs(d_def - d_nat), 4)
        rec["delta_agrees"] = abs(d_def - d_nat) <= max(
            0.10 * max(abs(d_def), abs(d_nat)), 1e-9)
        # If both deltas sit inside their own noise floor, the comparison is
        # uninformative in EITHER direction — that is not a disagreement.
        rec["delta_verdict"] = (
            "both-inside-noise" if both_noisy else
            "agrees" if rec["delta_agrees"] else "disagrees")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=3,
                    help="executions per (arch, arm) per round")
    ap.add_argument("--rounds", type=int, default=4,
                    help="rotation rounds. MUST be a multiple of 4 for the "
                         "sequence positions to balance exactly across the four "
                         "(arch, arm) combinations; anything else leaves a "
                         "residual drift bias, which is reported as "
                         "`mean_position` per arm")
    ap.add_argument("--kernels", default="layer_normalization/10_dynamic_16x512xHxW_HxW_HxW",
                    help="comma-separated category/case pairs")
    ap.add_argument("--out", default=os.path.join(HERE, "raw", "arch_e5_probe.json"))
    a = ap.parse_args()

    if a.rounds % 4:
        print(f"*** WARNING: --rounds {a.rounds} is not a multiple of 4.")
        print("    With 4 (arch, arm) combinations, only a multiple of 4 makes")
        print("    every combination occupy every sequence slot equally often.")
        print("    Anything else leaves drift bias in the delta -- the exact")
        print("    artifact this probe was rewritten to remove. Use 4 or 8.")

    if not os.path.isdir(LOCK):
        print(f"*** WARNING: GPU lock NOT held at {LOCK}")
        print("    This is a timing probe; without the lock another worker's")
        print("    nvcc can land entirely on one arch and fake a difference.")
        print("    Run via: mkdir -p benchmark2/.gpu-lock  (or run.sh exclusive)")

    pairs = []
    for spec in a.kernels.split(","):
        spec = spec.strip()
        if not spec:
            continue
        cat, _, case = spec.partition("/")
        pairs.append((cat, case))

    print(f"[arch-e5] {len(pairs)} kernel(s), reps={a.reps}, rounds={a.rounds}")
    t0 = time.time()
    recs = []
    for i, (cat, case) in enumerate(pairs, 1):
        print(f"  [{i}/{len(pairs)}] {cat}/{case} ...", flush=True)
        r = one_kernel(cat, case, a.reps, a.rounds)
        recs.append(r)
        if "error" in r:
            print(f"      ERROR: {r['error']}")
            continue
        for arch in ("default", "native"):
            on = r["arms"].get(f"{arch}/on", {})
            off = r["arms"].get(f"{arch}/off", {})
            wn = r["delta_within_noise"][arch]
            print(f"      {arch:8} ({on.get('nv_arch')}): "
                  f"on={on.get('marker_ms_median')}ms off={off.get('marker_ms_median')}ms "
                  f"delta={wn['delta_ms']}ms ({wn['delta_pct']}%)  "
                  f"spread={wn['arm_spread_pct']}%  "
                  f"within_noise={wn['within_noise']}")
            print(f"               wall on={on.get('wall_s_median')}s "
                  f"off={off.get('wall_s_median')}s   "
                  f"mean_position on={on.get('mean_position')} "
                  f"off={off.get('mean_position')}")
        print(f"      -> {r.get('delta_verdict')} "
              f"(abs diff between arches {r.get('delta_abs_diff_ms')}ms, "
              f"clock_drift {r.get('clock_drift_mhz')}MHz)")

    n_agree = sum(1 for r in recs if r.get("delta_verdict") == "agrees")
    n_dis = sum(1 for r in recs if r.get("delta_verdict") == "disagrees")
    n_noisy = sum(1 for r in recs if r.get("delta_verdict") == "both-inside-noise")
    n_err = sum(1 for r in recs if "error" in r)
    payload = {
        "question": "does -arch change what E5a measures (the checks-on/off delta)?",
        "why": "choreo defaults to nv_arch=sm_86 on sm_90 hardware; E5a is the "
               "only lane that EXECUTES, so it is the only one arch could move. "
               "E4 times nvcc (immune); E2/E3 are static (arch_sweep.py: "
               "309/311 label-only).",
        "jit_note": "cuobjdump on the default build shows sm_86.cubin x2 AND "
                    "sm_86.ptx x1, so the driver JITs at module load. Load is "
                    "process startup, which the kernel's own `Execution time` "
                    "marker excludes -- so JIT should not enter E5a's primary "
                    "timing source. wall_s is recorded alongside to test that.",
        "method": "all four (arch, arm) combinations are built up front and then "
                  "executed in a ROTATING order (each round shifts the sequence "
                  "by one position) so every combination occupies every slot "
                  "equally often. delta = on - off at the SAME arch, so arch is "
                  "common-mode; the test is whether the delta is stable.",
        "method_note": "A first cut blocked by arm (all `on` then all `off`) and "
                       "reported delta 19ms at sm_86 vs 173ms at sm_90a. That was "
                       "drift bias, not arch: this host idles at 345MHz with a "
                       "1755MHz max and persistence_mode Disabled, so clock ramp "
                       "trends monotonically across a blocked sequence and "
                       "whichever combination runs later absorbs it. run_e5.py "
                       "documents the same failure mode. Rotation plus "
                       "`mean_position` per arm is the defense; `clock_drift_mhz` "
                       "is the evidence.",
        "reps": a.reps,
        "rounds": a.rounds,
        "device": DEVICE,
        "lock_held": os.path.isdir(LOCK),
        "elapsed_s": round(time.time() - t0, 1),
        "n_kernels": len(recs),
        "n_errors": n_err,
        "n_delta_agrees": n_agree,
        "n_delta_disagrees": n_dis,
        "n_delta_both_inside_noise": n_noisy,
        "verdict_meaning": {
            "agrees": "both arches produce the same delta, and at least one "
                      "delta EXCEEDS its arms' own rep-to-rep spread, so the "
                      "delta is a measurement and it is arch-independent",
            "disagrees": "the two arches produce different deltas that exceed "
                         "the noise floor -- arch genuinely moves E5a's result",
            "both-inside-noise": "BOTH deltas are smaller than their own arms' "
                                 "rep-to-rep spread. This is NOT evidence that "
                                 "arch matters; it is evidence that the delta is "
                                 "not measurable at all on this host. The honest "
                                 "RQ3 statement is that residue is below the "
                                 "measurement floor.",
        },
        "verdict": (
            "arch DOES move E5a's delta -- re-run E5 at -arch=native"
            if n_dis > 0 else
            "arch does NOT change E5a's delta -- the lane is arch-independent in "
            "the quantity it reports"
            if n_agree > 0 and n_noisy == 0 else
            "INCONCLUSIVE ON ARCH, but the delta is below the noise floor: "
            f"{n_noisy}/{len(recs)} kernel(s) have BOTH deltas inside their own "
            "arms' rep-to-rep spread. Arch is not the problem -- the 345->1755MHz "
            "clock ramp is. E5a cannot resolve six entry guards against a ~2400ms "
            "kernel on this host; report residue as below the measurement floor "
            "rather than quoting a delta_pct."
            if n_noisy > 0 else "inconclusive"),
        "records": recs,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote {os.path.relpath(a.out, REPO)}  ({payload['elapsed_s']}s)")
    print(f"  agrees: {n_agree}   disagrees: {n_dis}   "
          f"both-inside-noise: {n_noisy}   errors: {n_err}")
    print(f"  VERDICT: {payload['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
