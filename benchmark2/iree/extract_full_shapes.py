#!/usr/bin/env python3
"""benchmark2/iree/extract_full_shapes.py — concrete per-shape dims.

Reads each provenance case's concrete dimensions from the choreo source
`#define` blocks (the numeric `__STATIC_SHAPE__`/unconditional values) and
maps them onto the operator signature's symbolic dims (from
benchmark2/settings/<category>.md). Emits:

  full_shapes.json  — per case: {category, case, defines, dims}
  sizes.py          — per-category SMALL / FULL / FULL_RAGGED (plan §2.4)

A dim with no numeric #define stays dynamic (`?`) — the case has no pinned
full size and is recorded as such.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent      # svn-artifacts/
CHOREO = ROOT / "benchmark" / "choreo"
SETTINGS = ROOT / "benchmark2" / "settings"
HERE = ROOT / "benchmark2" / "iree"
sys.path.insert(0, str(ROOT / "scripts"))
import gen_iree_cases as g  # noqa: E402  (shapes_for_category)

DEFINE_RE = re.compile(r"#define\s+([A-Za-z_][A-Za-z0-9_]*)\s+(\d+)")
DIMGRP = re.compile(r"\[([^\]]*)\]")
SIGROW = re.compile(
    r"^\|\s*\d+\s*\|\s*`([^`]+\.co)`\s*\|\s*`[0-9a-f]+`\s*\|\s*`(.*?)`\s*\|",
    re.MULTILINE)


def numeric_defines(co: Path) -> dict[str, int]:
    txt = co.read_text()
    out = {}
    for sym, val in DEFINE_RE.findall(txt):
        out[sym] = int(val)
    return out


def signature_symbols(sig: str) -> list[list[str]]:
    m = re.search(r"\(([^)]*)\)", sig)
    if not m:
        return []
    params = []
    for grp in DIMGRP.findall(m.group(1)):
        syms = [t.strip() for t in grp.split(",") if t.strip()]
        if syms:
            params.append(syms)
    return params


def concrete_dims(cat: str, stem: str, sig: str, defines: dict[str, int]):
    """Combine filename literals + signature symbols + #define values."""
    fname_dims = g.shapes_for_category(stem, cat)       # int / '?'
    sig_syms = signature_symbols(sig)                   # per-operand token lists
    out = []
    for oi, fd in enumerate(fname_dims):
        ss = sig_syms[oi] if oi < len(sig_syms) else []
        dims = []
        for i, d in enumerate(fd):
            if isinstance(d, int):
                dims.append(d)
                continue
            tok = ss[i] if i < len(ss) else "?"
            if tok.isdigit():
                dims.append(int(tok))
            elif tok in defines:
                dims.append(defines[tok])
            else:
                dims.append("?")
        out.append(dims)
    return out


def main():
    cases = {}
    for sf in sorted(SETTINGS.glob("*.md")):
        cat = sf.stem
        for row in SIGROW.finditer(sf.read_text()):
            co_path = Path(row.group(1))
            sig = row.group(2)
            case = co_path.name[:-3]
            co = CHOREO / cat / co_path.name
            defines = numeric_defines(co) if co.exists() else {}
            dims = concrete_dims(cat, case, sig, defines)
            cases[f"{cat}/{case}"] = {
                "category": cat, "case": case,
                "defines": defines, "dims": dims,
            }

    per_cat = {}
    for cat in sorted(p.stem for p in SETTINGS.glob("*.md")):
        rep = None
        for v in cases.values():
            if v["category"] == cat and v["case"].startswith("10_dynamic_"):
                rep = v
                break
        if rep is None:
            for v in cases.values():
                if v["category"] == cat:
                    rep = v
                    break
        per_cat[cat] = rep

    (HERE / "full_shapes.json").write_text(
        json.dumps({"cases": cases, "per_category_representative": per_cat},
                   indent=1) + "\n")

    # --- sizes.py: per-category SMALL / FULL / FULL_RAGGED (plan §2.4) -------
    def small_dim(d):
        return (d % 7) + 3 if isinstance(d, int) else 8

    def ragged(dims):
        out = [list(op) for op in dims]
        for op in out:
            for i in range(len(op)):
                if isinstance(op[i], int):
                    op[i] += (1 if i == len(op) - 1 else 0)
        return out

    sizes = {"SMALL": {}, "FULL": {}, "FULL_RAGGED": {}}
    for cat, rep in per_cat.items():
        if rep is None:
            continue
        full = rep["dims"]
        small = [[small_dim(d) for d in op] for op in full]
        sizes["FULL"][cat] = full
        sizes["SMALL"][cat] = small
        sizes["FULL_RAGGED"][cat] = ragged(full)
    (HERE / "sizes.py").write_text(
        "# Concrete SMALL / FULL / FULL_RAGGED shapes per category (plan §2.4).\n"
        "# Derived from the provenance .co #define blocks; see full_shapes.json.\n"
        "SMALL = " + repr(sizes["SMALL"]).replace('], ', '],\n        ') + "\n"
        "FULL = " + repr(sizes["FULL"]).replace('], ', '],\n       ') + "\n"
        "FULL_RAGGED = " + repr(sizes["FULL_RAGGED"]).replace('], ', '],\n              ') + "\n"
    )

    # --- manifest.csv: per-case mapping (category, case, choreo, iree, ...) ---
    lines = ["category,case_name,choreo_case,iree_case,expected,notes"]
    for v in cases.values():
        cat = v["category"]; case = v["case"]
        choreo = f"benchmark/choreo/{cat}/{case}.co"
        iree = f"benchmark2/iree/kernels/{cat}/{case}.mlir"
        lines.append(f"{cat},{case},{choreo},{iree},success,")
    (HERE / "manifest.csv").write_text("\n".join(lines) + "\n")

    dyn_cases = sum(1 for v in cases.values()
                    if any(d == "?" for op in v["dims"] for d in op))
    print(f"[extract_full_shapes] {len(cases)} cases; "
          f"{dyn_cases} still-dynamic dims; -> full_shapes.json, sizes.py, manifest.csv")


if __name__ == "__main__":
    main()
