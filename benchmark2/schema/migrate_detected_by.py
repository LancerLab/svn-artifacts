#!/usr/bin/env python3
"""Re-express a committed `mutants.jsonl` under the §9.6.1b measured vocabulary.

v1's `outcome` enum conflated an emitted check with a raw hang/crash -- both
were `runtime`. `specs/mutation-specs-v2.md` §9.6.1b rule 3 makes `runtime` mean
exactly `rt-check` (an emitted check the toolchain generated for the bug fired)
and re-expresses a bare hang, segfault, or abort as `never`, with the mechanism
in the new `detected_by` field. This is the deterministic, no-re-run correction
the spec prescribes: the mechanism is already recorded in `detail`, so `stage`
stays a total projection of `outcome` (`never -> none`).

Mapping (idempotent; a corpus from the fixed classifier is left as-is):

    detail TIMEOUT                       -> outcome=never, detected_by=hang
    detail SIGSEGV                       -> outcome=never, detected_by=segv
    detail `exit N`                      -> outcome=never, detected_by=nonzero-exit
    outcome=runtime, assert in detail    -> detected_by=gpu-assert
    outcome=runtime, otherwise           -> detected_by=rtv-assert
    outcome=compile                      -> detected_by=compile-error
    outcome=never                        -> detected_by=none
    outcome=n/a                          -> detected_by=n/a

usage: migrate_detected_by.py <mutants.jsonl> [more.jsonl ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: `stage` is a projection of `outcome` (see schema/records.py), not an
#: independent measurement; re-derive it so the total mapping cannot drift.
STAGE_FOR_OUTCOME = {
    "compile": "compile",
    "runtime": "runtime",
    "never": "none",
    "n/a": "none",
}

#: `outcome` values that a mechanism re-expresses, for the `distribution` string.
HANG_MECHANISMS = {"hang", "segv", "nonzero-exit"}


def mechanism(rec: dict) -> str:
    """The §9.6.1b mechanism of one record, from its own fields."""
    if rec.get("detected_by"):
        return str(rec["detected_by"])
    outcome = str(rec.get("outcome", ""))
    detail = str(rec.get("detail", "") or "")
    if outcome == "compile":
        return "compile-error"
    if outcome == "n/a":
        return "n/a"
    if outcome == "runtime":
        upper = detail.upper()
        if "TIMEOUT" in upper:
            return "hang"
        if "SIGSEGV" in upper:
            return "segv"
        if detail.startswith("exit "):
            return "nonzero-exit"
        # A device `cf.assert` surfaces as a per-thread `Assertion ... failed`
        # line; `_detail` truncates at 120 chars, so also accept the CUDA
        # block/thread prefix rather than requiring the word "Assertion".
        if any(m in detail for m in
               ("CUDA_ERROR_ASSERT", "Assertion", "block:", "thread:")):
            return "gpu-assert"
        return "rtv-assert"
    return "none"


def outcome_for(mechanism_name: str, prev: str) -> str:
    """The §9.6.1b outcome. Only an emitted check is `runtime` (rt-check)."""
    if mechanism_name in HANG_MECHANISMS:
        return "never"
    if mechanism_name in ("rtv-assert", "gpu-assert"):
        return "runtime"
    return prev


def migrate(path: Path) -> tuple[int, int]:
    """Rewrite one corpus in place. Returns (rows, reclassified)."""
    rows = [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]
    reclassified = 0
    for rec in rows:
        mech = mechanism(rec)
        outcome = outcome_for(mech, str(rec.get("outcome", "")))
        if outcome != rec.get("outcome"):
            reclassified += 1
            # `distribution` is `<n>x <outcome>/<manifest>`, a census of the
            # per-run verdicts; re-label the runs the mechanism re-expresses.
            dist = str(rec.get("distribution", ""))
            rec["distribution"] = dist.replace("runtime/", f"{outcome}/")
        rec["outcome"] = outcome
        rec["stage"] = STAGE_FOR_OUTCOME.get(outcome, "none")
        rec["detected_by"] = mech
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows)
                    + "\n")
    return len(rows), reclassified


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    for arg in argv:
        path = Path(arg)
        rows, reclassified = migrate(path)
        print(f"{path}: {rows} rows, {reclassified} reclassified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
