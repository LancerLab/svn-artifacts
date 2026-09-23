"""S12 over the *current* corpus: run compute-sanitizer memcheck on every
(category, family) pair present in raw/mutants.jsonl and upsert
raw/sanitizer.jsonl.

`driver.py sanitizer` walks the MUTANTS table, which the corpus outgrew (the
M3 mutation-only surfaces and the M4 iteration-space families, plus the carrier
rows added via add_families.py, are not all in the table). This script reads the
corpus itself so S12 covers exactly what E1 injected.

    python3 sanitizer_corpus.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import driver  # noqa: E402


def corpus_pairs(raw: Path):
    seen = []
    for line in (raw / "mutants.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        fam = int(r["mutant_id"].rsplit("-f", 1)[1])
        pair = (r["category"], fam)
        if pair not in seen:
            seen.append(pair)
    return seen


def main():
    raw = HERE / "raw"
    pairs = corpus_pairs(raw)
    for category, family in pairs:
        if category not in driver.CURRENT_TABLE:
            print(f"skip {category}-f{family}: not in MUTANTS table",
                  file=sys.stderr)
            continue
        rec = driver.run_mutant_sanitizer(category, family, raw)
        driver.emit(raw / "sanitizer.jsonl", rec)
        print(rec["mutant_id"], "flagged=" + rec["flagged"], rec["fault"],
              "exercised=" + rec["exercised"])
    print(f"S12: {len(pairs)} corpus pairs sanitized")


if __name__ == "__main__":
    main()
