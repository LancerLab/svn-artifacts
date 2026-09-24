"""Per-kernel --max-local-mem-capacity overrides (local-to-shared-migration W2).

raw/local_cap.json maps BASE kernel ids ("category/name") to
{"cap_bytes": int, "reason": str}. flags_for() returns the override flag for
that kernel, or [] when the kernel compiles at the compiler default. Mutants
inherit their base's entry: callers pass the base id (rec["category"] +
"/" + rec["case"]), never the mutant id. Every entry was measured by the
2026-09-24 default-cap probe (/tmp/probe_w2.jsonl); usage proved invariant
across the ledger/detector/oracle flag arms on the failing set.

CLI for shell consumers:
    python3 local_caps.py flags <category/name>     # prints the flag or nothing
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE = None


def _table():
    global _CACHE
    if _CACHE is None:
        with open(os.path.join(_HERE, "raw", "local_cap.json")) as f:
            _CACHE = json.load(f)
    return _CACHE


def cap_bytes_for(kernel_id):
    """kernel_id = 'category/name' (base id). Returns the int cap or None."""
    e = _table().get(kernel_id)
    if isinstance(e, dict) and "cap_bytes" in e:
        return int(e["cap_bytes"])
    return None


def flags_for(kernel_id):
    cap = cap_bytes_for(kernel_id)
    return ["--max-local-mem-capacity=%d" % cap] if cap else []


def kernel_id_from_path(path):
    """Derive the base id ('category/name') from a source path, else None.

    Recognises suite paths (benchmark/choreo/<cat>/<name>.co) and mutant paths
    (benchmark2/choreo/mutants/<M>/<cat>/<mutant>__<name>.co; mutants inherit
    their base's entry). Scratch copies (k.co in /tmp) are not derivable --
    callers with category/case in scope must pass kernel_id explicitly.
    """
    parts = os.path.normpath(path).split(os.sep)
    if not parts[-1].endswith(".co") or len(parts) < 3:
        return None
    stem = parts[-1][:-3]
    if "__" in stem:                                   # mutant: <m>__<base>.co
        return "%s/%s" % (parts[-2], stem.split("__", 1)[1])
    if parts[-3] == "choreo":                          # suite: choreo/<cat>/<n>.co
        return "%s/%s" % (parts[-2], stem)
    return None


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "flags":
        print(" ".join(flags_for(sys.argv[2])))
    else:
        sys.exit("usage: local_caps.py flags <category/name>")
