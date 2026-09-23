#!/usr/bin/env python3
"""Re-express committed corpora under the `mutation-specs-v2.md` §9.6.1b vocabulary.

Two changes, both idempotent:

* `manifest: "corrupts"` -> `"value-changing"`. §9.6.1b warns that the old
  ground-truth value collides with the `corrupt` *outcome* ("our mutation emitted
  illegal code"); the oracle property is "the output changed", so it is renamed.
* add the derived `measured` field (`avoid | corrupt | ct-check | rt-check |
  never`) to any record that carries `outcome`. It is computed from
  `outcome`/`detected_by`; the same mapping the classifier uses.

A `.jsonl` is rewritten one record per line (`sort_keys=True`, the `RecordWriter`
form). A `.json` that is a record or a list of records is rewritten structurally;
any other `.json` (an aggregate such as `stats.json`) gets a textual manifest
rename so its formatting is preserved. Dated archival snapshots are not touched.

usage: migrate_manifest_value.py <file.jsonl|file.json> [more ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MEASURED_BY_OUTCOME = {
    "n/a": "avoid",
    "compile": "ct-check",
    "runtime": "rt-check",
    "never": "never",
}


def _measured(rec: dict) -> str:
    if rec.get("measured"):
        return rec["measured"]
    if rec.get("detected_by") == "emitter-defect":
        return "corrupt"
    return MEASURED_BY_OUTCOME.get(rec.get("outcome", "never"), "never")


def _fix_str(s: str) -> str:
    return s.replace("corrupts", "value-changing")


def _walk(obj) -> None:
    if isinstance(obj, dict):
        if obj.get("manifest") == "corrupts":
            obj["manifest"] = "value-changing"
        for k, v in list(obj.items()):
            if isinstance(v, str):
                obj[k] = _fix_str(v)
            else:
                _walk(v)
        if "outcome" in obj and "manifest" in obj and "measured" not in obj:
            obj["measured"] = _measured(obj)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v)


def _is_record(obj) -> bool:
    return isinstance(obj, dict) and "outcome" in obj and "manifest" in obj


def migrate_jsonl(path: Path) -> int:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for r in rows:
        _walk(r)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return len(rows)


def migrate_json(path: Path) -> int:
    text = path.read_text()
    obj = json.loads(text)
    if isinstance(obj, list) and obj and _is_record(obj[0]):
        _walk(obj)
        path.write_text(json.dumps(obj, indent=2) + "\n")
        return len(obj)
    if _is_record(obj):
        _walk(obj)
        path.write_text(json.dumps(obj, indent=2) + "\n")
        return 1
    path.write_text(_fix_str(text))
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for arg in argv:
        p = Path(arg)
        if not p.exists():
            print(f"skip (missing): {p}")
            continue
        if p.suffix == ".jsonl":
            n = migrate_jsonl(p)
            print(f"{p}: {n} rows")
        else:
            n = migrate_json(p)
            print(f"{p}: {n} records")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
