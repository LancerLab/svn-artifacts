#!/usr/bin/env python3
"""DIAGNOSTIC (2026-09-09): does the NEW binary reject E5a's kernel launches?

WHY THIS EXISTS
---------------
croqtile commit 1fa4719 ("codegen: check launch errors after plain <<<>>> kernel
launches") documents that with `--max-local-mem-capacity=2000000` -- the CAP flag
pinned in EVERY choreo lane -- the dynamic-shape path sizes the per-thread local
arena to the full capacity, needing 466.9 GB on an H800 PCIe against 85 GB of
device memory, so the launch is REJECTED with cudaErrorInvalidValue. Before this
commit the rejection was SILENT: cudaDeviceSynchronize() succeeded when nothing
was enqueued, the harness still printed an "Execution time", and the process
exited 0. 36 of 153 dynamic benchmark cases were affected and all reported
success.

E5a was measured on the PRE-fix binary (680533f). All 14 of its kernels are
shape_class=dynamic. If those launches were being silently rejected, then E5a's
absolute milliseconds measured a kernel that NEVER RAN, and the whole E5a residue
A/B is void -- not noisy, void.

This script settles it. It rebuilds a sample of E5a's kernels with the CURRENT
binary (which now reports launch errors loudly) using E5a's exact pinned flags,
runs them, and greps for the rejection. It does NOT touch the artifact, the GPU
lock, or any lane output.

This is a correctness probe, not a timing measurement, so host contention does
not invalidate it -- we read the rc and the presence/absence of a launch error,
not the milliseconds.
"""
import os
import re
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_e5  # noqa: E402

# The exact flags E5a pins. FLAGS_ON is the checks-enabled arm; the launch
# rejection (if any) is a property of the generated code + CAP, not of the
# runtime checks, so one arm suffices to answer the question.
FLAGS = run_e5.FLAGS_ON

# A representative sample of E5a's 14 dynamic kernels: the largest, the
# smallest, and the two whose E5b mutants DID show the rejection (matmul,
# conv2d) so we can see whether the base kernel shares the mutants' fate.
SAMPLE = [
    ("layer_normalization", "10_dynamic_16x512xHxW_HxW_HxW"),
    ("relu", "10_dynamic_16x512xHxW_16x512xHxW"),
    ("matmul", "10_dynamic_128x1280_1280xN_128xN"),
    ("elemwise_add", "10_dynamic_16x512xHxW_16x512xHxW_16x512xHxW"),
    ("max_pool2d", "10_dynamic_32xCxHxW_32xCxHd5xWd5"),
]

REJECT = re.compile(r"cudaErrorInvalidValue|invalid argument|cudaLaunchKernel|"
                    r"launch (?:error|failed)|abend", re.I)
EXEC_TIME = re.compile(r"Execution time", re.I)


def main():
    import toolchain
    idt = toolchain.identity(refresh=True)
    print(f"binary  : {idt['binary']}  sha1={idt['binary_sha1_12']}  "
          f"mtime={toolchain._fmt(idt['binary_mtime'])}")
    print(f"version : {idt['version'][:12]} ({idt['version_basis']})")
    print(f"checkout: {idt['checkout_head'][:12]}")
    print(f"flags   : {' '.join(FLAGS)}")
    print(f"device  : {os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    print("=" * 78)

    workdir = "/tmp/diag_launch_reject"
    os.makedirs(workdir, exist_ok=True)
    results = []
    for cat, case in SAMPLE:
        d, err = run_e5.prep(cat, case, workdir)
        if err:
            print(f"{cat}/{case[:40]:<42} PREP FAIL: {err}")
            results.append({"kernel": f"{cat}/{case}", "prep_error": err})
            continue
        script, err = run_e5.build(d, FLAGS, "on")
        if err:
            print(f"{cat}/{case[:40]:<42} BUILD FAIL: {err[:120]}")
            results.append({"kernel": f"{cat}/{case}", "build_error": err[:300]})
            continue
        rc, so, se, wall = run_e5.run_exe(script, {"CUDA_VISIBLE_DEVICES": "0"})
        blob = (so or "") + "\n" + (se or "")
        rejected = bool(REJECT.search(blob))
        printed_time = bool(EXEC_TIME.search(blob))
        verdict = ("LAUNCH REJECTED" if rejected
                   else ("ran (printed Execution time)" if printed_time
                         else f"ran? rc={rc}"))
        print(f"{cat}/{case[:40]:<42} rc={rc:<4} wall={wall:6.2f}s  "
              f"reject={rejected!s:<5} exec_time={printed_time!s:<5}  {verdict}")
        # Show the first rejection line if any, so the evidence is on the record.
        if rejected:
            for ln in blob.splitlines():
                if REJECT.search(ln):
                    print(f"      | {ln.strip()[:160]}")
                    break
        results.append({
            "kernel": f"{cat}/{case}", "rc": rc, "wall_s": round(wall, 3),
            "launch_rejected": rejected, "printed_execution_time": printed_time,
            "verdict": verdict,
        })

    print("=" * 78)
    n_rej = sum(1 for r in results if r.get("launch_rejected"))
    n_ok = sum(1 for r in results
               if r.get("printed_execution_time") and not r.get("launch_rejected"))
    print(f"SUMMARY: {n_rej}/{len(results)} LAUNCH-REJECTED, "
          f"{n_ok}/{len(results)} ran and printed Execution time")
    if n_rej == 0:
        print("  -> E5a's kernels LAUNCH FINE on the new binary. The silent-"
              "rejection class does not include them; E5a's absolute ms are "
              "real work, and the E5a residue A/B stands.")
    elif n_rej == len(results):
        print("  -> EVERY sampled E5a kernel is launch-rejected. E5a measured "
              "kernels that never ran on the pre-fix binary. E5a is VOID and "
              "must be re-measured on the new binary.")
    else:
        print("  -> MIXED. Some E5a kernels are affected. Identify exactly "
              "which and re-measure only those; do not quote E5a wholesale.")
    with open("/tmp/diag_launch_reject.json", "w") as f:
        json.dump({"toolchain_version": idt["version"],
                   "binary_sha1_12": idt["binary_sha1_12"],
                   "flags": FLAGS, "results": results,
                   "n_rejected": n_rej, "n_ran": n_ok}, f, indent=1)
    print("  wrote /tmp/diag_launch_reject.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
