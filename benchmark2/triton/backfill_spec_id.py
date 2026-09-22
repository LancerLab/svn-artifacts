"""Backfill `spec_id` into `raw/mutants.jsonl` from `driver.SPEC_ID`.

One-off metadata fix (no GPU): the committed rows predate the `spec_id` field,
so their 27 M1 + 2 M3 instances cannot be attributed to a family and the
worklist marks every family `unattributed(no spec_id)`. The mapping lives in
`driver.py`, so the value written here is the same one a re-run of `minimal`
would emit.

    python3 backfill_spec_id.py      # rewrite raw/mutants.jsonl in place
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from driver import spec_id_for  # noqa: E402


def family_of(mutant_id: str):
    m = re.search(r"-f(\d+)$", mutant_id)
    return int(m.group(1)) if m else None


def main() -> None:
    path = HERE / "raw" / "mutants.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]
    changed = 0
    for r in rows:
        sid = spec_id_for(r["category"], r["class"],
                          family_of(r["mutant_id"]))
        if r.get("spec_id") != sid:
            r["spec_id"] = sid
            changed += 1
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{changed}/{len(rows)} row(s) updated -> {path}")


if __name__ == "__main__":
    main()
