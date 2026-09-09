#!/usr/bin/env python3
"""E4 — compile cost (choreo only). Feeds S10 → RQ4.

Plan §6.1: "RQ4 (0.6% median compile overhead) stands. One-time price,
host-CPU wall-clock, not GPU-contended; record under quiet host load."

--------------------------------------------------------------------------
WHAT IS MEASURABLE, AND WHAT IS NOT  (feasibility probe, step 0 of the template)
--------------------------------------------------------------------------
The obvious design — compile with checks on, compile with checks off, report the
delta — CANNOT BE RUN on the pinned build. Measured, not assumed:

  1. The assessor cannot be disabled. `-rtc` controls only which *runtime
     assertions are emitted into the generated CUDA*; the assessment pipeline
     itself always runs. `-rtc=none`, `--disable-runtime-check` and
     `-zero-cost` all produce a byte-identical ledger.

  2. The front-end time is identical across `-rtc` levels. On
     layer_normalization/3_attention_32xNx512x64_64_64, 30 interleaved reps:
         -rtc=entry  median 21.70 ms  (min 20.83, stdev 0.33)
         -rtc=none   median 21.68 ms  (min 20.78, stdev 0.35)
         delta +0.11%  — inside the noise floor, and the sign is not stable.
     So there is no "checks-off" compile to subtract. Any number produced that
     way would be noise dressed as a measurement.

  3. nvcc dominates and is noisy. Full compile-link is 9.1–24.6 s with ~10%
     rep-to-rep spread on identical input; the choreo front-end is 8–22 ms.

What IS separable is the front-end itself: the time choreo spends parsing,
assessing and emitting CUDA, against the nvcc time that ANY tile DSL pays to
compile the resulting CUDA. That is the honest reading of "compile overhead" —
the cost choreo adds to a compilation that would otherwise be nvcc's alone.

    compile_overhead_pct = 100 * t_frontend / (t_frontend + t_nvcc)

`t_nvcc` is measured by running the generated script's own `--compile-link`
path, so it uses choreo's exact nvcc flags rather than a reconstruction.

--------------------------------------------------------------------------
DISCREPANCY WITH THE PAPER — FLAGGED, NOT SILENTLY RESOLVED
--------------------------------------------------------------------------
The probe gives 0.06%–0.24% per kernel (median ~0.13%), i.e. BELOW RQ4's
claimed 0.6%. The two are not the same quantity: RQ4's figure comes from the
prior benchmark's 151-kernel CSV under a definition this script cannot
reproduce (the checks-off arm does not exist). This script therefore reports
its own definition explicitly on every record (`definition` field) and prints a
loud notice. The coordinator must decide which number the paper carries; this
lane does not overwrite RQ4 by fiat.

--------------------------------------------------------------------------
`bucket` — the schema names the field but defines no enum. Defined here:
    "<0.5%" | "0.5-1%" | "1-2%" | "2-5%" | ">=5%"
RQ4's 0.6% falls in "0.5-1%", so the bucket lets the integrator check the claim
without depending on an exact median that shifts with host load.

--------------------------------------------------------------------------
GPU / EXCLUSIVITY — and why this lane takes the lock anyway
--------------------------------------------------------------------------
§12.6's prose calls E4 "host-CPU time, not GPU-contended", so the first cut of
this script took no lock and hardcoded `exclusive=false`. That was wrong, and
not for a GPU reason: §12.6 rule 2 says "the `qc` worker rejects any `cost`
record with `exclusive=false`", and `cost` is E4's record type, emitted by no
other lane. Read literally, an unlocked E4 produces 15 records qc must throw
away, leaving S10/RQ4 with no data at all.

So: run E4 under `benchmark2/.gpu-lock` (run.sh:cmd_e4 wraps it in
`exclusive`), and derive `exclusive` from the lock's ACTUAL presence via
gpuinfo.lock_held() rather than hardcoding it. Stamping true without holding
the lock would be a fabricated run condition — worse than an inadmissible
record, because qc would pass it.

The real threat to this lane is host contention, not GPU contention: nvcc
saturates every core, and a wall-clock measurement taken while another lane
compiles is drift, not overhead. `host_busy()` therefore gates the run and
`--force` overrides it with every record stamped `host_contended`.

FEEDS  S10 (median compile-overhead %, choreo) → RQ4.
"""

import argparse
import json
import os
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

RAW = os.path.join(HERE, "raw")
OUT = os.path.join(RAW, "e4_compile_cost.json")
WORKDIR = os.path.join(RAW, "e4_logs")

TOOLCHAIN = "choreo"
CHOREO = os.path.join(REPO, "croqtile", "build-release", "choreo")
SUITE = os.path.join(REPO, "benchmark", "choreo")

CAP = "--max-local-mem-capacity=2000000"
# Front-end only: -es emits the target source and skips nvcc entirely.
FE_FLAGS = ["-gs", "-t", "cute", "-kt", CAP, "-es"]
# Full script generation (no -es) so --compile-link can then run nvcc.
GEN_FLAGS = ["-gs", "-t", "cute", "-kt", CAP]

T_FRONTEND = 120
T_NVCC = 1800

DEFINITION = (
    "compile_overhead_pct = 100 * t_choreo_frontend / "
    "(t_choreo_frontend + t_nvcc_compile_link); frontend = choreo -gs -t cute "
    "-kt <k> -es (parse + assess + emit CUDA, nvcc skipped); nvcc = the "
    "generated script's own --compile-link path with choreo's exact flags. "
    "The assessor cannot be disabled on this build (-rtc only gates emitted "
    "assertions; -rtc=none/-zero-cost/--disable-runtime-check give a "
    "byte-identical ledger), so a checks-on/checks-off delta is not "
    "obtainable and is NOT what this number reports."
)

BUCKETS = [
    (0.5, "<0.5%"),
    (1.0, "0.5-1%"),
    (2.0, "1-2%"),
    (5.0, "2-5%"),
]


def bucket_of(pct):
    for hi, label in BUCKETS:
        if pct < hi:
            return label
    return ">=5%"


def prep(category, case, workdir):
    """Copy the kernel and its local headers into a scratch dir.

    The generated CUDA resolves `#include "common.h"` via `-I<dir of the .co>`,
    so the category's local headers MUST sit next to the copy or nvcc dies with
    "fatal error: common.h: No such file or directory". Same trap E1 hit.
    """
    d = os.path.join(workdir, f"{category}_{case}")
    os.makedirs(d, exist_ok=True)
    src = os.path.join(SUITE, category, case + ".co")
    dst = os.path.join(d, "k.co")
    if not os.path.exists(src):
        return None, f"source missing: {src}"
    with open(src, "rb") as f:
        blob = f.read()
    with open(dst, "wb") as f:
        f.write(blob)
    for h in sorted(os.listdir(os.path.join(SUITE, category))):
        if h.endswith((".h", ".hpp")):
            with open(os.path.join(SUITE, category, h), "rb") as f:
                hb = f.read()
            with open(os.path.join(d, h), "wb") as f:
                f.write(hb)
    return d, None


def time_frontend(d, reps):
    """choreo front-end only (-es): parse + assess + emit, no nvcc."""
    out = os.path.join(d, "k_fe.sh")
    samples = []
    for _ in range(reps):
        t = time.perf_counter()
        r = subprocess.run([CHOREO] + FE_FLAGS + [os.path.join(d, "k.co"),
                                                  "-o", out],
                           capture_output=True, timeout=T_FRONTEND, cwd=REPO)
        samples.append(time.perf_counter() - t)
        if r.returncode != 0:
            return None, samples, r.returncode, r.stderr.decode("utf8", "replace")[-400:]
    return statistics.median(samples), samples, 0, None


def time_nvcc(d, reps):
    """The generated script's own --compile-link: choreo's exact nvcc flags."""
    out = os.path.join(d, "k.sh")
    r = subprocess.run([CHOREO] + GEN_FLAGS + [os.path.join(d, "k.co"),
                                               "-o", out],
                       capture_output=True, timeout=T_FRONTEND, cwd=REPO)
    if r.returncode != 0 or not os.path.exists(out):
        return None, [], r.returncode, "script generation failed"
    samples = []
    for _ in range(reps):
        t = time.perf_counter()
        p = subprocess.run(["bash", out, "--compile-link"],
                           capture_output=True, timeout=T_NVCC, cwd=REPO)
        samples.append(time.perf_counter() - t)
        if p.returncode != 0:
            return None, samples, p.returncode, \
                p.stderr.decode("utf8", "replace")[-400:]
    return statistics.median(samples), samples, 0, None


def measure_one(category, case, workdir, fe_reps, nvcc_reps):
    d, err = prep(category, case, workdir)
    rec = {
        "toolchain": TOOLCHAIN,
        "category": category,
        "kernel_id": f"{category}/{case}",
        "kernel_hash": run_e2.sha1_12(os.path.join(SUITE, category, case + ".co")),
        "settings_hash": run_e2.settings_hash(category),
        "fe_reps": fe_reps,
        "nvcc_reps": nvcc_reps,
    }
    if d is None:
        rec["ok"] = False
        rec["error"] = err
        return rec

    fe, fe_s, rc, msg = time_frontend(d, fe_reps)
    rec["frontend_s_samples"] = [round(x, 5) for x in fe_s]
    if fe is None:
        rec["ok"] = False
        rec["error"] = f"frontend rc={rc}: {msg}"
        return rec

    nv, nv_s, rc, msg = time_nvcc(d, nvcc_reps)
    rec["nvcc_s_samples"] = [round(x, 3) for x in nv_s]
    if nv is None:
        rec["ok"] = False
        rec["error"] = f"nvcc rc={rc}: {msg}"
        return rec

    rec["ok"] = True
    rec["frontend_s"] = round(fe, 5)
    rec["nvcc_s"] = round(nv, 4)
    rec["total_s"] = round(fe + nv, 4)
    rec["compile_overhead_pct"] = round(100.0 * fe / (fe + nv), 4)
    rec["overhead_vs_nvcc_pct"] = round(100.0 * fe / nv, 4)
    rec["bucket"] = bucket_of(rec["compile_overhead_pct"])
    # min is recorded alongside the median: nvcc has ~10% rep-to-rep spread, so
    # a reader judging stability needs both, not just the point estimate.
    rec["frontend_s_min"] = round(min(fe_s), 5)
    rec["nvcc_s_min"] = round(min(nv_s), 4)
    rec["nvcc_s_spread_pct"] = round(
        100.0 * (max(nv_s) - min(nv_s)) / statistics.median(nv_s), 2) \
        if len(nv_s) > 1 else 0.0
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=1,
                    help="kept at 1 by default: this is a WALL-CLOCK lane and "
                         "parallel nvcc runs contend for host CPU, inflating "
                         "every sample. Raise only if the host is otherwise "
                         "idle and you accept the contamination.")
    ap.add_argument("--fe-reps", type=int, default=5,
                    help="front-end reps (cheap, ~20 ms each); median reported")
    ap.add_argument("--nvcc-reps", type=int, default=3,
                    help="nvcc reps (expensive, ~10-25 s each); median reported")
    ap.add_argument("--only", default="", help="restrict to one category")
    ap.add_argument("--limit", type=int, default=0,
                    help="measure at most N kernels (0 = all)")
    ap.add_argument("--size", default="small", choices=["small", "full"])
    ap.add_argument("--device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    ap.add_argument("--workdir", default=WORKDIR)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--keep-logs", action="store_true",
                    help="keep the per-kernel scratch dirs (default: removed)")
    ap.add_argument("--force", action="store_true",
                    help="measure even though the host is contended; every "
                         "record is stamped `host_contended` so the integrator "
                         "cannot mistake it for a clean sample")
    a = ap.parse_args()

    a.workdir = os.path.abspath(a.workdir)
    a.out = os.path.abspath(a.out)
    os.makedirs(a.workdir, exist_ok=True)

    if not os.path.exists(CHOREO):
        print(f"ERROR: no choreo binary at {CHOREO}", file=sys.stderr)
        print("       run `benchmark2/choreo/run.sh setup` first", file=sys.stderr)
        return 2

    ver = run_e2.toolchain_version() if hasattr(run_e2, "toolchain_version") else None

    kernels = run_e2.all_kernels()
    if a.only:
        kernels = [k for k in kernels if k[0] == a.only]
    if a.size == "small":
        # One static + one dynamic case per category: the two shape classes are
        # where compile cost can differ, and S10 is reported per category.
        seen = {}
        picked = []
        for cat, case in kernels:
            s = seen.setdefault(cat, {"static": None, "dynamic": None})
            try:
                dyn, _, _ = run_e2.is_dynamic(os.path.join(SUITE, cat, case + ".co"))
            except Exception:
                continue
            key = "dynamic" if dyn else "static"
            if s[key] is None:
                s[key] = (cat, case)
                picked.append((cat, case))
        kernels = picked
    if a.limit:
        kernels = kernels[:a.limit]

    print(f"[choreo] E4: compile cost over {len(kernels)} kernels "
          f"(size={a.size}, jobs={a.jobs}, fe_reps={a.fe_reps}, "
          f"nvcc_reps={a.nvcc_reps})")
    print(f"[choreo] definition: {DEFINITION}")

    # §12.6 rule 3: record the run conditions, not just the number. E4 is a
    # wall-clock lane, so the host state matters more than the GPU state, but
    # both are stamped so a reviewer can see what the sample was taken under.
    excl = gpuinfo.lock_held()
    conds = gpuinfo.run_conditions(a.device, exclusive=excl)
    busy = gpuinfo.host_busy()
    clk = conds["gpu_snapshot"].get("gpus", {}).get(
        str(a.device), {}).get("clocks.sm")
    print(f"[choreo] device={a.device} exclusive={excl} sm_clock={clk}MHz "
          f"lock={'held' if excl else 'NOT HELD'}")
    if not excl:
        print("*** exclusive=false: qc REJECTS `cost` records carrying it "
              "(§12.6 rule 2). ***")
        print("    Run this lane via `run.sh e4`, which takes .gpu-lock, or")
        print("    create the lock yourself. These records will not survive qc.")
    if busy:
        print(f"*** HOST CONTENDED: competing process(es) alive: {busy} ***")
        print("    E4 is a wall-clock lane; nvcc from another worker makes")
        print("    every sample below drift rather than overhead.")
        if not a.force:
            print("    (pass --force to measure anyway and stamp "
                  "`host_contended` on every record)")
            return 3

    est = len(kernels) * (a.fe_reps * 0.02 + a.nvcc_reps * 12 + 12)
    print(f"[choreo] estimated wall clock: ~{est/60:.0f} min")

    t0 = time.time()
    records = []
    for i, (cat, case) in enumerate(kernels, 1):
        rec = measure_one(cat, case, a.workdir, a.fe_reps, a.nvcc_reps)
        records.append(rec)
        if rec.get("ok"):
            print(f"  [{i:>3}/{len(kernels)}] {rec['kernel_id'][:56]:<58} "
                  f"fe={rec['frontend_s']*1000:6.1f}ms "
                  f"nvcc={rec['nvcc_s']:6.2f}s "
                  f"overhead={rec['compile_overhead_pct']:6.3f}% "
                  f"[{rec['bucket']}]")
        else:
            print(f"  [{i:>3}/{len(kernels)}] {cat}/{case} FAILED: "
                  f"{rec.get('error', '')[:110]}")
        if not a.keep_logs:
            d = os.path.join(a.workdir, f"{cat}_{case}")
            subprocess.run(["rm", "-rf", d])

    ok = [r for r in records if r.get("ok")]
    bad = [r for r in records if not r.get("ok")]

    # ---------------------------------------------------- per-category rollup ----
    by_cat = {}
    for r in ok:
        c = by_cat.setdefault(r["category"], [])
        c.append(r["compile_overhead_pct"])

    cost_records = []
    for cat in sorted(by_cat):
        pcts = by_cat[cat]
        med = statistics.median(pcts)
        cost_records.append({
            "toolchain": TOOLCHAIN,
            "category": cat,
            "compile_overhead_pct": round(med, 4),
            "bucket": bucket_of(med),
            # --- provenance / cross-cutting (schema: every record needs these) ---
            "settings_hash": next(r["settings_hash"] for r in ok
                                  if r["category"] == cat),
            "kernel_hash": None,          # per-category aggregate: no single kernel
            "toolchain_version": ver,
            "size": a.size,
            "gpu_device": conds["gpu_device"],
            "exclusive": conds["exclusive"],
            "gpu_snapshot": conds["gpu_snapshot"],
            "host_contended": busy,
            "n_kernels": len(pcts),
            "min_pct": round(min(pcts), 4),
            "max_pct": round(max(pcts), 4),
            "mean_pct": round(statistics.mean(pcts), 4),
            "definition": DEFINITION,
        })

    all_pcts = [r["compile_overhead_pct"] for r in ok]
    grand = statistics.median(all_pcts) if all_pcts else 0.0

    out = {
        "toolchain": TOOLCHAIN,
        "toolchain_version": ver,
        "produced_by": "e4",
        "size": a.size,
        "gpu_device": conds["gpu_device"],
        "exclusive": conds["exclusive"],
        "gpu_snapshot": conds["gpu_snapshot"],
        "host_contended": busy,
        "elapsed_s": round(time.time() - t0, 1),
        "definition": DEFINITION,
        "fe_reps": a.fe_reps,
        "nvcc_reps": a.nvcc_reps,
        "jobs": a.jobs,
        "grand_median_compile_overhead_pct": round(grand, 4),
        "grand_bucket": bucket_of(grand),
        "records": cost_records,
        "per_kernel": sorted(ok, key=lambda r: r["kernel_id"]),
        "failures": bad,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {os.path.relpath(a.out, REPO)}  "
          f"({round(time.time()-t0,1)}s)")

    # ------------------------------------------------------------- report ----
    print("\n=== S10 compile overhead per category (median) ===")
    print(f"{'category':<22}{'n':>4}{'median%':>10}{'min%':>9}{'max%':>9}  bucket")
    for r in cost_records:
        print(f"{r['category']:<22}{r['n_kernels']:>4}"
              f"{r['compile_overhead_pct']:>10.3f}{r['min_pct']:>9.3f}"
              f"{r['max_pct']:>9.3f}  {r['bucket']}")
    print(f"{'-'*22}{'-'*4}{'-'*10}")
    print(f"{'GRAND MEDIAN':<22}{len(all_pcts):>4}{grand:>10.3f}"
          f"{'':>9}{'':>9}  {bucket_of(grand)}")

    if bad:
        print(f"\n*** {len(bad)} kernel(s) FAILED to measure (excluded from "
              f"the median, not counted as zero cost) ***")
        for r in bad[:15]:
            print(f"    {r['kernel_id']}: {r.get('error','')[:120]}")

    print("\n*** DISCREPANCY NOTICE — READ BEFORE QUOTING THIS NUMBER ***")
    print(f"  This lane measures {grand:.3f}% (median, {len(all_pcts)} kernels).")
    print("  RQ4 in the paper claims 0.6% median compile overhead.")
    print("  These are NOT the same quantity. RQ4's figure comes from the prior")
    print("  benchmark under a checks-on/checks-off definition that CANNOT be")
    print("  reproduced on this build: the assessor is not disableable, and the")
    print("  front-end time is identical across -rtc levels (+0.11% over 30")
    print("  interleaved reps, sign unstable). This script reports the only")
    print("  separable cost — choreo's front-end against the nvcc baseline any")
    print("  tile DSL pays — and states that definition on every record.")
    print("  The coordinator decides which number the paper carries.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
