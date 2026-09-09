#!/usr/bin/env python3
"""Mutation application for the benchmark2 MLIR lanes.

A mutation is applied by **perturbing the composition parameters** (the `Case`
dims), never by editing emitted IR text. This matters for the numbers: a
`sed`-style substitution that fails to match silently leaves a plain type error,
which is indistinguishable from a genuine detection.

Specs M2.1-M2.4 are pure shape perturbations, so they are expressed here as a new
`Case`. Spec M2.5 (partial / duplicate write) is *structural* -- it changes the
write pattern rather than a type -- so it is carried as a `Structural` value and
handled by the emitter.

Two invariants are enforced here because they directly affect the reported
counts:

* **No two specs may produce the same perturbation.** M2.2 and M2.4 both concern
  the output extent, so they perturb *different* axes and are checked for
  distinctness.
* **No mutation may be a silent no-op.** A perturbation landing on a dynamic axis
  cannot change the declared `?`, so it rebinds the concrete value in `dyn`
  instead. Anything that still changes nothing raises, because specs §7.1
  discards `noop` mutants and that would silently shrink the class below N=40.

The reference output is always computed from the **original** case, which is what
makes the §7 oracle meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import compose as C


@dataclass(frozen=True)
class Structural:
    """A mutation that cannot be expressed as a shape perturbation.

    `kind` is one of:
      * `partial-write`   -- only a prefix tile of the output is written; the tail
                             keeps its uninitialized value.
      * `duplicate-write` -- a tile is written twice, the second time at an
                             overlapping offset.
      * M1 kinds (`drop-mask`, `off-by-one`, `negative-index`,
        `transposed-stride`, `offset-overrun`, `zero-stride`) -- index/stride
        perturbations on the low-level surface.
    """

    kind: str
    tile: int = 0  # extent of the written tile along `axis`
    axis: int = 0
    offset: int = 0  # second write offset, for duplicate-write


def apply(case: C.Case, mut: C.Mutation) -> tuple[C.Case, Structural | None]:
    """Return (mutated case, structural directive or None)."""
    dims = dict(case.dims)
    dyn = dict(case.dyn)
    structural: Structural | None = None
    cat = case.category

    def bump(op: str, axis: int, by: int = 1) -> None:
        """Grow `op`'s extent at `axis`, handling the dynamic case."""
        d = list(dims[op])
        if d[axis] is None:
            key = f"{op}.{axis}"
            dyn[key] = dyn[key] + by
        else:
            d[axis] = d[axis] + by
            dims[op] = tuple(d)

    if mut.klass == "M2":
        if mut.spec == 1:
            # "caller passes scale (or a secondary operand) of the wrong leading
            # extent". The secondary operand differs per category. For
            # `elemwise_add` the secondary operand is `rhs`, and the defect is on
            # its LEADING axis -- M2.3 perturbs the same operand's TRAILING axis,
            # so the two stay distinct defects rather than one counted twice.
            op = {"matmul": "rhs", "concat": "b",
                  "layer_normalization": "scale",
                  "elemwise_add": "rhs"}.get(cat)
            if op is None:
                raise NotApplicable(
                    f"M2.1: {cat!r} has no secondary operand to mis-size"
                )
            bump(op, 0)

        elif mut.spec == 2:
            # "DMA source/destination extent disagreement". The linalg analogue is
            # an output whose extent no longer agrees with what the *inputs*
            # imply. Perturb a trailing axis so this stays distinct from M2.4,
            # which perturbs the leading axis.
            out = dims["out"]
            axis = len(out) - 1
            bump("out", axis)

        elif mut.spec == 3:
            # "binary op on mismatched shapes" -- two operands of one binary op
            # disagree. matmul has no elementwise binary op, so it is genuinely
            # n/a there rather than being forced onto some other defect.
            # `elemwise_add` is this spec's natural home (mutation-specs.md §5
            # lists it as the level-2 M2 category for exactly this reason): its
            # `lhs`/`rhs` feed one `arith.addf`.
            if cat == "layer_normalization":
                bump("bias", 0)
            elif cat == "concat":
                bump("a", len(dims["a"]) - 1)
            elif cat == "elemwise_add":
                bump("rhs", len(dims["rhs"]) - 1)
            else:
                raise NotApplicable(
                    f"M2.3: {cat!r} composes no binary elementwise op"
                )

        elif mut.spec == 4:
            # "output tensor declared with a wrong leading extent".
            bump("out", 0)

        elif mut.spec == 5:
            # "partial write (omitted tail tile) / duplicate write (overlapping
            # tile)" -- structural, handled by the emitter.
            out_r = _concrete(case, "out")
            axis = len(out_r) - 1
            extent = out_r[axis]
            if extent < 2:
                raise NotApplicable(
                    f"M2.5: output axis {axis} has extent {extent}; a partial or "
                    f"overlapping tile needs at least 2"
                )
            if mut.variant == "duplicate":
                # Write the first tile again at offset 1, so tile 0 and tile 1
                # overlap and one row is clobbered by its neighbour.
                structural = Structural(
                    kind="duplicate-write", axis=axis, tile=extent - 1, offset=1
                )
            else:
                structural = Structural(
                    kind="partial-write", axis=axis, tile=extent - 1
                )

        else:
            raise KeyError(f"unknown M2 spec {mut.spec}")

    elif mut.klass == "M1":
        # M1 is the mlir-low lane's class; it perturbs indices and strides rather
        # than declared shapes. Those are carried structurally because the
        # low-level emitter builds the access expression itself.
        structural = _m1_structural(case, mut)

    else:
        raise KeyError(f"unsupported mutation class {mut.klass!r}")

    mutated = replace(
        case, dims=dims, dyn=dyn, case_id=f"{case.case_id}_{mut.mutant_id}"
    )
    if structural is None and not _differs(case, mutated):
        raise AssertionError(
            f"{mut.mutant_id} on {cat} changed nothing -- would be recorded as a "
            f"`noop` false success (specs §7.1) and silently shrink N"
        )
    return mutated, structural


class NotApplicable(Exception):
    """The surface cannot express this defect on this category.

    Recorded as `outcome=n/a` (mutation-specs.md §6), never re-balanced and never
    counted as a detection.
    """


def _concrete(case: C.Case, name: str) -> tuple[int, ...]:
    return tuple(
        d if d is not None else case.dyn[f"{name}.{k}"]
        for k, d in enumerate(case.dims[name])
    )


def _differs(a: C.Case, b: C.Case) -> bool:
    """True if the mutation changed any declared extent or dynamic binding."""
    if a.dims != b.dims or a.dyn != b.dyn:
        return True
    return any(_concrete(a, k) != _concrete(b, k) for k in a.dims)


def _m1_primary(case: C.Case) -> str:
    """Name of the operand whose indices M1 perturbs.

    Most low-level categories carry a single `inp`, but `layer_normalization`
    spells its primary operand `lhs` (with `scale`/`bias` as secondary). Hardcoding
    `inp` made M1 on layer_norm raise `KeyError` instead of measuring anything --
    and §5's minimal set requires M1 on layer_norm.
    """
    for name in ("inp", "lhs"):
        if name in case.dims:
            return name
    raise KeyError(f"M1: {case.category!r} has no primary input operand")


def _m1_structural(case: C.Case, mut: C.Mutation) -> Structural:
    """M1 defects are index/stride perturbations on the low-level surface."""
    extents = _concrete(case, _m1_primary(case))
    last = len(extents) - 1
    if mut.spec == 1:
        # dropped boundary mask: the guard on the indexed access is omitted.
        return Structural(kind="drop-mask", axis=last)
    if mut.spec == 2:
        return Structural(kind="off-by-one", axis=last, offset=1)
    if mut.spec == 3:
        return Structural(kind="negative-index", axis=last, offset=-1)
    if mut.spec == 4:
        return Structural(kind="transposed-stride", axis=last)
    if mut.spec == 5:
        return Structural(kind="offset-overrun", axis=last, offset=extents[last])
    if mut.spec == 6:
        return Structural(kind="zero-stride", axis=last)
    raise KeyError(f"unknown M1 spec {mut.spec}")
