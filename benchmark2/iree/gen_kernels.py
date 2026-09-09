#!/usr/bin/env python3
"""benchmark2/iree/gen_kernels.py — settings-driven IREE kernel composer.

Worker-owned derivation (§2.2/§2.5): reads ONLY benchmark2/settings/<cat>.md
(operator contracts — shapes/semantics/provenance), never the benchmark source,
and emits one linalg-on-tensors module per case under
benchmark2/iree/kernels/<cat>/<stem>.mlir using the same per-operator body
templates as scripts/gen_iree_cases.py (the established IREE authoring path).

Every emitted kernel is fingerprinted; provenance (settings_hash, kernel_hash)
is recorded in kernels/manifest.jsonl and carried onto every raw record by the
harness (schema/record-schema.json cross-cutting fields).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent          # svn-artifacts/
sys.path.insert(0, str(ROOT / "scripts"))
import gen_iree_cases as g  # noqa: E402  (shapes_for_category, GENERATORS)

LANE = ROOT / "benchmark2" / "iree"
SETTINGS_DIR = ROOT / "benchmark2" / "settings"
OUT_DIR = LANE / "kernels"
MANIFEST = OUT_DIR / "manifest.jsonl"

ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*`([^`]+\.co)`\s*\|\s*`([0-9a-f]+)`\s*\|\s*`(.*?)`\s*\|", re.MULTILINE)

DIM_GRP = re.compile(r"\[([^\]]*)\]")


def dims_from_signature(sig: str):
    """Fallback: derive operand dims from the settings operator signature
    (used when the case stem carries no shape info, e.g. bench_relu)."""
    m = re.search(r"\(([^)]*)\)", sig)
    if not m:
        return None
    out = []
    for dg in DIM_GRP.findall(m.group(1)):
        dims = []
        for tok in dg.split(","):
            tok = tok.strip()
            if not tok:
                continue
            dims.append("?" if not tok.isdigit() else int(tok))
        if dims:
            out.append(dims)
    return out or None


def settings_hash(category: str) -> str:
    data = (SETTINGS_DIR / f"{category}.md").read_text().encode()
    return hashlib.sha1(data).hexdigest()[:12]


def case_rows(category: str):
    text = (SETTINGS_DIR / f"{category}.md").read_text()
    for m in ROW.finditer(text):
        yield m.group(2), m.group(3), m.group(4)


def generate(category: str, stem: str, sig: str, gen_fn):
    shapes = g.shapes_for_category(stem, category)
    if not shapes:
        shapes = dims_from_signature(sig)
    if not shapes:
        return None
    code = gen_fn(stem, shapes)
    if code is None:
        return None
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=None)
    ap.add_argument("--force", action="store_true", help="regenerate existing")
    args = ap.parse_args()

    categories = sorted(p.stem for p in SETTINGS_DIR.glob("*.md"))
    if args.category:
        categories = [args.category]

    existing = {}
    if MANIFEST.exists() and not args.force:
        for line in MANIFEST.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            existing[rec["category"] + "/" + rec["stem"]] = rec

    total = ok = skipped = 0
    with MANIFEST.open("a") if not args.force else MANIFEST.open("w") as mf:
        if args.force:
            mf.write("")
        for cat in categories:
            gen_fn = g.GENERATORS.get(cat)
            if gen_fn is None:
                print(f"[iree/gen_kernels] no generator for {cat}")
                continue
            sh = settings_hash(cat)
            for src_path, src_sha, sig in case_rows(cat):
                total += 1
                stem = Path(src_path).stem
                key = f"{cat}/{stem}"
                if key in existing and not args.force:
                    ok += 1
                    continue
                code = generate(cat, stem, sig, gen_fn)
                out = OUT_DIR / cat / f"{stem}.mlir"
                out.parent.mkdir(parents=True, exist_ok=True)
                if code is None:
                    skipped += 1
                    print(f"[iree/gen_kernels] SKIP {key} (unsupported shapes)")
                    continue
                out.write_text(code)
                kh = hashlib.sha1(code.encode()).hexdigest()[:12]
                mf.write(json.dumps({
                    "toolchain": "iree", "category": cat, "stem": stem,
                    "settings_hash": sh, "settings_source_sha1": src_sha,
                    "kernel_hash": kh,
                }) + "\n")
                ok += 1
    print(f"[iree/gen_kernels] {ok}/{total} emitted ({skipped} skipped) -> {OUT_DIR}")


if __name__ == "__main__":
    main()
