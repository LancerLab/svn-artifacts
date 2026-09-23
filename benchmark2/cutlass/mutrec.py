#!/usr/bin/env python3
"""mutrec.py — the ONE place this lane builds a mutant record.

The lane records used to carry a private shape (`{spec_id, category, mutation,
outcome, detail}` with outcomes `ct-check`/`rt-check`/`unchecked`/`noop`). That
shape is not the contract: `schema/record-schema.json` v2.1 defines a mutant
record as the base set

    toolchain, category, class, paper_category, mutant_id, level, outcome,
    stage, manifest

plus the ported set

    spec_id, path_class, prohibition, applicable, spec_version

and constrains `outcome`/`stage`/`manifest`/`paper_category`/`path_class`/
`prohibition`/`applicable`/`spec_version` to fixed vocabularies. A private shape
validates as v1 by default (`records.spec_version_of`), so the omission is
silent -- exactly the failure `corpus.declared-files` exists to catch.

This module is the single writer. It maps this lane's §9.6 outcome words onto
the canonical ones, derives `stage` from `schema.records.stage_for` (the one
definition, so it cannot drift), and validates every record before returning it
-- a record this lane cannot validate is a bug in the caller, not something to
commit and explain later.

Outcome mapping (§9.6 -> schema):
    ct-check  -> compile  (nvcc/CuTe rejects the mutant)
    rt-check  -> runtime  (the mutant builds, the launch/run faults)
    unchecked -> never    (builds and runs, produces different output silently)
    noop      -> n/a      (the mutation has no representable effect)
"""

import os
import sys

LANE_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH2 = os.path.dirname(LANE_DIR)                  # benchmark2/
if BENCH2 not in sys.path:
    sys.path.insert(0, BENCH2)
from schema import records as REC                               # noqa: E402

TOOLCHAIN = "cutlass"
TOOLCHAIN_VERSION = "CUTLASS v4.2.1 / CUDA 12.9"
SPEC_VERSION = "v2.1"

# Lane outcome word -> canonical `outcome`.
OUTCOME = {"ct-check": "compile", "rt-check": "runtime",
           "unchecked": "never", "noop": "n/a"}

# §9.6.1b measured vocabulary: the finest statement the data supports. A
# `compile` is a ct-check unless audited to the emitter (then `corrupt`); a
# bare crash carries no emitted check, so it is `never`.
MEASURED = {"compile": "ct-check", "runtime": "rt-check",
            "never": "never", "n/a": "avoid"}

# spec -> paper category. M3.2/M3.4 are stride (a descriptor/stride overshoot);
# the rest of the M3 battery is a shape/divisibility overshoot (dim-mismatch).
# Every M4 spec is a loop/stride bound, so stride. `L` is the launch-status path
# class, which is a hardware limit (hw).
PAPER_CATEGORY = {
    "M3.1": "dim-mismatch",
    "M3.2": "stride",
    "M3.3": "dim-mismatch",
    "M3.4": "stride",
    "M3.5": "dim-mismatch",
    "M3.6": "dim-mismatch",
    "M3.7": "dim-mismatch",
    "M3.8": "dim-mismatch",
    "M3.11": "dim-mismatch",
    "M3.12": "dim-mismatch",
    "M3.13": "dim-mismatch",
    "M3.14": "dim-mismatch",
}
M4_PAPER = "stride"
L_PAPER = "hw"

# M1 (element access) maps to the published {oob, stride, wrong-shape} (see
# class-axis.json's `published_taxonomy` note): a bound/offset/tile-coordinate
# overshoot is `oob`, a step defect is `stride`, and an index that names the
# wrong dimension/arity is `wrong-shape`.
M1_PAPER = {
    "M1.2": "oob",           # p#n off-by-one
    "M1.4": "stride",        # transposed / non-contiguous stride
    "M1.9": "oob",           # base offset without shrinking the extent
    "M1.11": "wrong-shape",  # read-after-write aliasing overlap
    "M1.12": "wrong-shape",  # wrong loop variable for a dimension
    "M1.14": "oob",          # tile coordinate over/underflow
    "M1.15": "wrong-shape",  # index >= rank
    "M1.19": "oob",          # narrow index carrier
}


# M1.6/M1.7 were re-homed to M4 (families M4-d/M4-e, see method_taxonomy):
# they are realisations of the iteration-validity class, so their records must
# carry class M4, not the prefix's M1.
REHOMED_TO_M4 = {"M1.6", "M1.7"}


def class_of(spec_id):
    """The mutation class of a spec id. `L` is a path class of M3, not a class
    of its own: a launch-status defect is an M3 (hardware-constraint) defect
    whose path class happens to be `L`."""
    if spec_id.startswith("M4") or spec_id in REHOMED_TO_M4:
        return "M4"
    if spec_id.startswith("M3") or spec_id.startswith("L"):
        return "M3"
    if spec_id.startswith("M1"):
        return "M1"
    if spec_id.startswith("M2"):
        return "M2"
    raise ValueError(f"unclassifiable spec id {spec_id!r}")


def paper_category(spec_id):
    if spec_id in PAPER_CATEGORY:
        return PAPER_CATEGORY[spec_id]
    if spec_id in REHOMED_TO_M4:
        return M4_PAPER
    if spec_id in M1_PAPER:
        return M1_PAPER[spec_id]
    if spec_id.startswith("M4"):
        return M4_PAPER
    if spec_id.startswith("L"):
        return L_PAPER
    raise ValueError(f"no paper category for {spec_id!r}")


def make_record(*, spec_id, category, mutation, outcome, path_class,
                manifest, prohibition="", applicable=True, mutant_id,
                level="1", detail="", kernel_hash="", settings_hash="",
                arch="", note=""):
    """Build and validate a v2.1 mutant record. Raises on a bad record."""
    if outcome not in OUTCOME:
        raise ValueError(f"unknown lane outcome {outcome!r}")
    o = OUTCOME[outcome]
    rec = {
        "toolchain": TOOLCHAIN,
        "toolchain_version": TOOLCHAIN_VERSION,
        "category": category,
        "class": class_of(spec_id),
        "paper_category": paper_category(spec_id),
        "mutant_id": mutant_id,
        "level": level,
        "outcome": o,
        "stage": REC.stage_for(o),
        "manifest": manifest,
        "measured": MEASURED.get(o, "never"),
        "spec_id": spec_id,
        "path_class": path_class,
        "prohibition": prohibition,
        "applicable": "true" if applicable else "false",
        "spec_version": SPEC_VERSION,
        "mutation": mutation,
        "detail": detail,
    }
    for k, v in (("kernel_hash", kernel_hash), ("settings_hash", settings_hash),
                 ("arch", arch), ("note", note)):
        if v:
            rec[k] = v
    errs = REC.validate(rec, "mutant")
    if errs:
        raise RuntimeError(
            f"mutant record {mutant_id!r} is not a valid v2.1 record: {errs}")
    return rec
