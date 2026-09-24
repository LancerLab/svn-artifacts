"""probe_compile.py — re-run this lane's `compile` mutants and preserve the raw
child output as evidence for the ct-check classification.

Why this exists: `driver.run_mutant` classifies a crashed child as `compile`
(ct-check) when its output carries a compile-error marker. The original record
kept only the constant string "jit: child reported crash", so the 56 ct-checks
in `raw/mutants.jsonl` could not be audited from the corpus (§9.6.1b rule 2).

Run with the triton venv, from anywhere:

    python triton/probe_compile.py            # re-probe, write raw/compile_probe/
    python triton/probe_compile.py --update   # also rewrite the 56 record details

Every mutant that no longer shows a real Triton exception is reported and, with
`--update`, downgraded to `never` (rule 3) so the corpus cannot overclaim.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from driver import COMPILE_ERR_MARKERS, _diag_text  # noqa: E402

PROBE = ROOT / "raw" / "compile_probe"


def main() -> int:
    update = "--update" in sys.argv[1:]
    PROBE.mkdir(parents=True, exist_ok=True)
    path = ROOT / "raw" / "mutants.jsonl"
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    comp = [r for r in recs if r.get("outcome") == "compile"]
    downgraded, summary = [], []
    for i, r in enumerate(sorted(comp, key=lambda x: x["mutant_id"]), 1):
        mid = r["mutant_id"]
        mod, fam = mid.rsplit("-f", 1)
        try:
            p = subprocess.run(
                [sys.executable, str(ROOT / "mutants" / f"{mod}.py"),
                 "--family", fam, "--size", "full"],
                capture_output=True, text=True, timeout=180, cwd=str(ROOT))
            out = p.stdout + p.stderr
        except subprocess.TimeoutExpired:
            summary.append([mid, r["spec_id"], "TIMEOUT", ""])
            downgraded.append(mid)
            print(f"[{i:2}] {mid:16} {r['spec_id']:7} TIMEOUT")
            continue
        (PROBE / f"{mid}.err").write_text(out)
        matched = any(k in out for k in COMPILE_ERR_MARKERS)
        diag = _diag_text(out) if matched else ""
        summary.append([mid, r["spec_id"], "compile" if matched else "NONE",
                        diag])
        if not matched:
            downgraded.append(mid)
        if matched and update:
            r["detail"] = "jit: " + diag
        elif not matched and update:
            r["outcome"], r["stage"] = "never", "none"
            r["measured"] = "never"
            r["detail"] = "child crash (no compile diagnostic): " + diag
        print(f"[{i:2}] {mid:16} {r['spec_id']:7} "
              f"{'compile' if matched else 'NONE':7} {diag[:80]}")
    (PROBE / "_summary.json").write_text(json.dumps(summary, indent=2))
    if update:
        path.write_text("".join(json.dumps(x) + "\n" for x in recs))
        print(f"updated {len(comp) - len(downgraded)}/{len(comp)} details in {path}")
    if downgraded:
        print("NOT a ct-check (rule 3 -> never):", ", ".join(downgraded))
        return 1
    print(f"all {len(comp)} compile mutants show a real Triton diagnostic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
