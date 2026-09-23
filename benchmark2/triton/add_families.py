"""One-off: measure and upsert a NAMED subset of (category, family) mutants.

`driver.py minimal` re-runs the whole MUTANTS table, which would overwrite the
existing corpus. This script runs only the pairs given on the command line and
upserts them, so it can extend the corpus without touching rows already there.

    python3 add_families.py relu:8 relu:9 softmax:8 ...

Used for the M1-b (M1.8) / M1-c (M1.9) second realisations.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import driver  # noqa: E402


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: add_families.py category:family[:size] [...]")
    raw = HERE / "raw"
    for spec in sys.argv[1:]:
        cat, fam, *rest = spec.rsplit(":", 2)
        size = rest[0] if rest else "full"
        rec = driver.run_mutant(cat, int(fam), raw, size=size)
        driver.emit(raw / "mutants.jsonl", rec)
        print(rec["mutant_id"], rec["outcome"], rec["manifest"],
              rec["detail"][:80])


if __name__ == "__main__":
    main()
