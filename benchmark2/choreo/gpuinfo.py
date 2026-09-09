#!/usr/bin/env python3
"""gpuinfo — shared GPU run-condition helpers for the choreo lanes.

Exists because three separate scripts had grown their own copy of "is the lock
held / is the host busy", and plan §12.6 rule 3 needs a fourth thing none of
them had:

    3. Record `gpu_device`, `exclusive∈{true,false}`, and `nvidia-smi` snapshot
       (util/mem/clock) in the `kernel`/`cost` records so a reviewer can see the
       run conditions.

A reviewer cannot judge a timing number without knowing the clock it ran at.
This is not decoration: on this machine device 0 idles at 1755 MHz (max) while
device 1 idles at 345 MHz, so the same kernel on the two devices is not
comparable, and an E5a Δ measured during a clock ramp is drift, not overhead.

--------------------------------------------------------------------------
EXCLUSIVITY — the rule that decides whether a record survives qc
--------------------------------------------------------------------------
§12.6 rule 2: "`e5` (and any timing) runs only under an exclusivity lock
(`benchmark2/.gpu-lock`) ... and the run records `exclusive=true`. The `qc`
worker rejects any `cost` record with `exclusive=false`."

Note what that does and does not say. The *prose* scopes the lock to E5 and
calls E4 "host-CPU time, not GPU-contended". The *rule* rejects `cost` records
— and `cost` is E4's record type, produced by no other lane. Read literally,
an E4 run that stamps `exclusive=false` produces records qc must throw away,
which would leave S10/RQ4 with no data at all.

Resolution adopted here: **take the lock for E4 too.** It costs nothing (E4 is
the only lane running at that moment in `cmd_all`'s order, and the lock is
advisory between our own lanes), and it makes the record admissible under the
rule as written. `exclusive` is derived from the lock's actual presence, never
hardcoded — stamping `true` without holding it would be a lie, and
`lock_held()` is the single source of truth for both lanes.

FEEDS  run_e4.py, run_e5.py, collect.py (provenance).
"""

import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))          # benchmark2/choreo
B2 = os.path.dirname(HERE)                                 # benchmark2/

LOCK = os.path.join(B2, ".gpu-lock")

# Processes whose presence means a timing sample is contaminated. nvcc/cicc/
# ptxas saturate every core; compute-sanitizer serializes kernels and perturbs
# every co-resident process (§12.6 table, row 4).
NOISE_PROCS = {
    "nvcc", "cicc", "ptxas", "choreo", "compute-sanitizer", "memcheck",
    "ninja", "make", "cc1plus",
}

SMI_FIELDS = [
    "index", "name", "utilization.gpu", "memory.used", "memory.total",
    "clocks.sm", "clocks.max.sm", "temperature.gpu", "persistence_mode",
]


def lock_held():
    """Is the GPU lock present? The only admissible source of `exclusive`.

    Stamping exclusive=true without holding the lock is a fabricated run
    condition, which is worse than an inadmissible record: qc would pass it.
    """
    return os.path.isdir(LOCK)


def lock_info():
    """Who holds the lock, for a diagnostic message. '' if unreadable."""
    p = os.path.join(LOCK, "info")
    try:
        with open(p) as f:
            return " ".join(f.read().split())
    except OSError:
        return ""


def host_busy():
    """Sorted unique names of competing compile/run processes on this host.

    Empty means quiet. WHY THIS EXISTS: a first E5a cut reported checks-on 25%
    FASTER than checks-off — not a result, but a concurrent E4 run saturating
    every core with nvcc. A timing lane cannot distinguish "the guard is free"
    from "the host was stolen" after the fact, so contamination is detected up
    front and reported rather than discovered in the numbers.

    LIMITATION, stated rather than hidden: `ps -eo comm=` cannot tell our own
    child processes from someone else's. Callers must therefore check BEFORE
    spawning any nvcc of their own, which is how both lanes use it.
    """
    try:
        p = subprocess.run(["ps", "-eo", "comm="], capture_output=True,
                           text=True, timeout=15)
    except Exception:
        return []
    hits = set()
    for line in p.stdout.splitlines():
        base = os.path.basename(line.strip())
        if base in NOISE_PROCS:
            hits.add(base)
    return sorted(hits)


def smi_snapshot(device=None):
    """nvidia-smi run-conditions snapshot (§12.6 rule 3).

    Returns a dict for the requested device (or all devices if device is None),
    plus `compute_apps` — the PIDs currently holding the device, which is the
    direct evidence for or against exclusivity. Returns {} if nvidia-smi is
    unavailable, so a lane still produces records on a machine without it
    rather than dying; the absence is then visible as an empty field.
    """
    try:
        q = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(SMI_FIELDS)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
    except Exception:
        return {}
    if q.returncode != 0:
        return {}

    gpus = {}
    for line in q.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(SMI_FIELDS):
            continue
        d = dict(zip(SMI_FIELDS, parts))
        idx = d.get("index")
        for k in ("utilization.gpu", "memory.used", "memory.total",
                  "clocks.sm", "clocks.max.sm", "temperature.gpu"):
            try:
                d[k] = int(d[k])
            except (ValueError, TypeError, KeyError):
                pass
        gpus[idx] = d

    apps = []
    try:
        a = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        if a.returncode == 0:
            for line in a.stdout.strip().splitlines():
                if line.strip():
                    apps.append(line.strip())
    except Exception:
        pass

    if device is None:
        return {"gpus": gpus, "compute_apps": apps}
    return {"gpus": {str(device): gpus.get(str(device), {})},
            "compute_apps": apps}


def device_clock(device):
    """SM clock (MHz) of one device, or None. Convenience for a report line."""
    s = smi_snapshot(device)
    g = s.get("gpus", {}).get(str(device), {})
    return g.get("clocks.sm")


def compact_snapshot(device):
    """The three numbers a timing record actually needs, flat.

    `smi_snapshot()` returns the full nested structure, which is right for a
    per-run header but too heavy to repeat on every one of ~300 records. This
    keeps clock/util/mem — the fields §12.6 rule 3 names — as scalars.

    Clock is the one that matters. On this machine device 0 idles at 1755 MHz
    and device 1 at 345 MHz, and a device ramps under load, so two samples of
    the same kernel can differ by the clock they happened to run at. Recording
    it per sample is what lets a reviewer tell guard overhead from a clock ramp
    after the fact — which is exactly the failure mode that produced E5a's
    implausible negative deltas.
    """
    s = smi_snapshot(device)
    g = s.get("gpus", {}).get(str(device), {})
    return {
        "sm_clock_mhz": g.get("clocks.sm"),
        "sm_clock_max_mhz": g.get("clocks.max.sm"),
        "gpu_util_pct": g.get("utilization.gpu"),
        "mem_used_mib": g.get("memory.used"),
        "temp_c": g.get("temperature.gpu"),
        "compute_apps": s.get("compute_apps", []),
    }


def run_conditions(device, exclusive=None):
    """The three §12.6 rule-3 fields, as one stampable dict.

    `exclusive` defaults to the lock's actual state. Callers that know better
    (a lane that deliberately shares the GPU for correctness work) pass it
    explicitly — but a TIMING lane must not, because then the lock is the
    authority and the record cannot overstate its own conditions.
    """
    if exclusive is None:
        exclusive = lock_held()
    return {
        "gpu_device": str(device),
        "exclusive": bool(exclusive),
        "gpu_snapshot": smi_snapshot(device),
    }
