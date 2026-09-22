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
      * `pad-swap`        -- the pad_low/pad_high amounts are swapped, keeping
                             the padded total but moving the data (M2.15).
      * `reshape-contiguous` -- a strided sub-span is reread with stride 1
                             (M2.18).
      * `reshape-short`   -- the runtime flatten extent is stale, so the result
                             is silently shortened (M2.17).
      * `broadcast-one`   -- a rank-1 broadcast extent is declared 1 and indexed
                             with a constant, so one element fills every position
                             (M2.8).
      * `broadcast-msb`   -- the higher-rank operand's leading broadcast extent
                             is made 2 and the second slice is read (M2.21).
      * M1 kinds (`drop-mask`, `off-by-one`, `negative-index`,
        `transposed-stride`, `offset-overrun`, `zero-stride`,
        `overlap-write`, `broadcast-index`, `tile-coord`, `narrow-carrier`) --
        index/stride perturbations on the low-level surface.
    """

    kind: str
    tile: int = 0  # extent of the written tile along `axis`
    axis: int = 0
    offset: int = 0  # second write offset, for duplicate-write
    perm: tuple[int, ...] = ()  # replacement permutation, for M2.10


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

    def swap(op: str, i: int, j: int) -> None:
        """Exchange `op`'s extents at axes `i` and `j`.

        A permutation, unlike `bump`, moves a declared extent rather than
        changing it, so it must carry the *concrete* value across when one of
        the two axes is dynamic. The declared `?` travels with its axis; the
        binding under `dyn` follows the concrete value to its new axis.
        """
        d = list(dims[op])
        ci = d[i] if d[i] is not None else dyn.get(f"{op}.{i}")
        cj = d[j] if d[j] is not None else dyn.get(f"{op}.{j}")
        d[i], d[j] = d[j], d[i]
        dims[op] = tuple(d)
        if d[i] is None and cj is not None:
            dyn[f"{op}.{i}"] = cj
        if d[j] is None and ci is not None:
            dyn[f"{op}.{j}"] = ci

    def drop_axis(op: str, axis: int) -> None:
        """Remove `op`'s extent at `axis` (family M2-c, M2.7).

        A reduced-rank view: the operand loses a dimension, so its rank no
        longer matches the op's contract. Dynamic bindings beyond the dropped
        axis shift down with their axis.
        """
        d = list(dims[op])
        dims[op] = tuple(x for k, x in enumerate(d) if k != axis)
        for k in range(len(d)):
            if k == axis:
                continue
            key = f"{op}.{k}"
            if key in dyn:
                nk = k - 1 if k > axis else k
                dyn[f"{op}.{nk}"] = dyn.pop(key)

    def merge_last_two(op: str) -> None:
        """Fold `op`'s final two extents into one (family M2-c, M2.16).

        The element count is preserved while the rank and split change, so the
        operand holds the same number of elements but no longer matches the
        expected rank. A dynamic trailing pair is folded into a concrete
        product and its stale bindings dropped.
        """
        d = list(dims[op])
        i = len(d) - 2
        ci = d[i] if d[i] is not None else dyn[f"{op}.{i}"]
        cj = d[i + 1] if d[i + 1] is not None else dyn[f"{op}.{i + 1}"]
        dims[op] = tuple(d[:i] + [ci * cj])
        for k in range(i, len(d)):
            dyn.pop(f"{op}.{k}", None)

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
            # which perturbs the leading axis. Output-only, so it would apply to
            # any category; it is held to the core set so the rank/axis variants
            # added to lift M2-d/g/h do not re-open the saturated M2-a battery.
            if cat not in C.CORE_M2_CATS:
                raise NotApplicable(
                    f"M2.2: {cat!r} is a mutation-only variant of another "
                    f"family's surface")
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
            # "output tensor declared with a wrong leading extent". Output-only,
            # held to the core set for the same reason as M2.2.
            if cat not in C.CORE_M2_CATS:
                raise NotApplicable(
                    f"M2.4: {cat!r} is a mutation-only variant of another "
                    f"family's surface")
            bump("out", 0)

        elif mut.spec == 5:
            # "partial write (omitted tail tile) / duplicate write (overlapping
            # tile)" -- structural, handled by the emitter. Output-only, held to
            # the core set so M2-f stays byte-identical.
            if cat not in C.CORE_M2_CATS:
                raise NotApplicable(
                    f"M2.5: {cat!r} is a mutation-only variant of another "
                    f"family's surface")
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

        elif mut.spec == 6:
            # "two extents transposed" (v2.1 family M2-b): a permutation of one
            # operand's extents. The pair is per category so it neither collides
            # with M2.9's leading pair on `concat` nor repeats an axis M2.1-M2.4
            # already perturbs.
            op, pair = {
                "matmul": ("lhs", (0, 1)),
                "concat": ("b", (1, 2)),
                "layer_normalization": ("lhs", (0, 2)),
                "elemwise_add": ("lhs", (0, 2)),
                "softmax": ("inp", (0, 1)),
            }.get(cat, (None, None))
            if op is None:
                raise NotApplicable(f"M2.6: {cat!r} is not an M2 category")
            swap(op, *pair)

        elif mut.spec == 9:
            # "batch/group dimension swapped" (v2.1 family M2-b): the leading
            # pair. A rank-2 operand has only one pair, so a batch/group swap
            # there is indistinguishable from M2.6 and would duplicate it; only
            # `concat`'s 4D `b` carries a batch/group axis distinct from the
            # concatenation axis, so the other categories are honestly n/a.
            if cat != "concat":
                raise NotApplicable(
                    f"M2.9: {cat!r} has no batch/group axis distinct from M2.6"
                )
            swap("b", 0, 1)

        elif mut.spec == 7:
            # "reduced-rank view" (v2.1 family M2-c): a dimension is dropped
            # from a declared operand, so its rank no longer matches the op.
            op = {"matmul": "lhs", "concat": "b",
                  "layer_normalization": "lhs", "elemwise_add": "lhs",
                  "softmax": "inp"}.get(cat)
            if op is None:
                raise NotApplicable(f"M2.7: {cat!r} is not an M2 category")
            drop_axis(op, len(dims[op]) - 1)

        elif mut.spec == 16:
            # "same element count, different rank/split" (v2.1 family M2-c):
            # the final two extents are folded together. Distinct from M2.7,
            # which drops an axis outright, because the count is preserved.
            op = {"matmul": "lhs", "concat": "b",
                  "layer_normalization": "lhs", "elemwise_add": "lhs",
                  "softmax": "inp"}.get(cat)
            if op is None:
                raise NotApplicable(f"M2.16: {cat!r} is not an M2 category")
            if len(dims[op]) < 2:
                raise NotApplicable(f"M2.16: {cat!r} operand has rank < 2")
            merge_last_two(op)

        elif mut.spec == 10:
            # "transpose permutation on a SQUARE operand" (v2.1 family M2-e):
            # every extent agrees, so only the memory order can be wrong. Only
            # the square transpose composes this; on a non-square transpose a
            # wrong permutation would change the output shape and be caught
            # trivially, which is not this defect.
            if cat not in C.TRANSPOSE_SQUARE_CATS:
                raise NotApplicable(
                    f"M2.10: {cat!r} has no square operand whose permutation "
                    f"can be changed with extents intact")
            structural = Structural(
                kind="perm", perm=tuple(range(len(dims["inp"]))))

        elif mut.spec == 13:
            # "shape-equal / layout-unequal" (v2.1 family M2-e): two equal
            # extents are swapped, so every shape stays legal while the indexing
            # map -- and therefore the element mapping -- changes.
            op = {"matmul": "lhs", "concat": "a",
                  "layer_normalization": "lhs", "elemwise_add": "lhs",
                  "softmax": "inp", "transpose_square": "inp",
                  "transpose_cube": "inp"}.get(cat)
            if op is None:
                raise NotApplicable(f"M2.13: {cat!r} is not an M2 category")
            pair = _equal_pair(_concrete(case, op))
            if pair is None:
                raise NotApplicable(
                    f"M2.13: {cat!r} {op!r} has no two equal extents; swapping "
                    f"unequal ones would change the shape (that is M2.6)")
            structural = Structural(kind="layout-unequal", perm=pair)

        elif mut.spec == 15:
            # "pad_low <-> pad_high swapped" (v2.1 family M2-g): the pad total is
            # preserved, so every extent agrees and the sum-only check is blind;
            # only the placement of the data moves. Only the `pad` kernel carries
            # a pad amount to swap, so the other categories are honestly n/a.
            if cat not in C.PAD_CATS:
                raise NotApplicable(
                    f"M2.15: {cat!r} composes no padded span whose low/high "
                    f"placement can be swapped")
            structural = Structural(kind="pad-swap")

        elif mut.spec == 17:
            # "runtime-shaped flatten with a stale extent" (v2.1 family M2-h):
            # the element-count check is skipped when BOTH source and result are
            # runtime-shaped, so a wrong runtime extent survives. A static operand
            # is not runtime-shaped, so there is nothing to skip and the defect is
            # honestly not expressible there.
            if cat not in C.RESHAPE_CATS:
                raise NotApplicable(
                    f"M2.17: {cat!r} composes no runtime-shaped flatten")
            if not case.is_dynamic():
                raise NotApplicable(
                    "M2.17: static operand -- the element-count check is not "
                    "skipped, so a stale extent is caught (not this defect)")
            structural = Structural(kind="reshape-short")

        elif mut.spec == 18:
            # "reshape a non-contiguous span" (v2.1 family M2-h): the strided
            # sub-span is reread with stride 1, so the flatten linearises the
            # wrong elements. Only the reshape kernel composes such a span.
            if cat not in C.RESHAPE_CATS:
                raise NotApplicable(
                    f"M2.18: {cat!r} composes no non-contiguous sub-span")
            structural = Structural(kind="reshape-contiguous")

        elif mut.spec == 21:
            # "MSB broadcast extent neither 1 nor equal" (v2.1 family M2-d): the
            # higher-rank operand's LEADING extent is made 2 and slice 1 is read.
            # The rank-unequal compatibility walk compares only the trailing dims
            # (semacheck.cpp:543-595), so this passes. Only the broadcast kernel
            # composes a rank-unequal pair whose leading extent is a broadcast.
            if cat not in C.BROADCAST_CATS:
                raise NotApplicable(
                    f"M2.21: {cat!r} composes no rank-unequal broadcast whose "
                    f"leading extent can be mis-declared")
            structural = Structural(kind="broadcast-msb")

        elif mut.spec == 8:
            # "a broadcast extent that must be N is 1" (v2.1 family M2-d): the
            # rank-1 secondary operand is declared (1,) and indexed with a
            # constant, so its single element is spread over every position. Only
            # the broadcast kernel carries a rank-1 operand that broadcasts.
            if cat not in C.BROADCAST_CATS:
                raise NotApplicable(
                    f"M2.8: {cat!r} composes no broadcast extent that can be set "
                    f"to 1")
            structural = Structural(kind="broadcast-one")

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


def _equal_pair(r: tuple[int, ...]) -> tuple[int, int] | None:
    """First two equal extents of `r`, or None. Used by M2.13, which is only
    shape-preserving when the swapped extents are equal."""
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            if r[i] == r[j]:
                return (i, j)
    return None


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
    if mut.spec == 11:
        # Read-after-write aliasing overlap (family M1-h). The written region is
        # shrunk by one, so successive tiles map onto the same live slots and
        # clobber each other. `tile` carries the shrunk region size; the emitter
        # wraps the store index modulo it.
        return Structural(kind="overlap-write", axis=last,
                          tile=max(1, extents[last] - 1))
    if mut.spec == 12:
        # Wrong loop variable used for a dimension (family M1-d). The emitter
        # substitutes a different induction variable for the axis's own index.
        if last < 1:
            raise KeyError(f"M1.12: {case.category!r} rank < 2, nothing to reuse")
        return Structural(kind="broadcast-index", axis=last)
    if mut.spec == 14:
        # Tile-coordinate over/underflow (family M1-e). `tile` carries the axis
        # extent; the emitter applies `_tile_for` to pick the tile size and
        # advances the read index by that whole tile.
        return Structural(kind="tile-coord", axis=last, tile=extents[last])
    if mut.spec == 19:
        # Narrow index carrier (family M1-f). The kernel carries the innermost
        # element index in an integer narrower than the axis it walks, so a
        # coordinate the buffer admits wraps to a negative offset. `tile` carries
        # the carrier bit width; the emitter casts the read index through `i16`
        # and back. The axis extent must exceed what that width can span, or the
        # mutant would be inert -- the lane binds this spec only to the wide
        # carrier kernels (`compose.M1_CARRIER_CATS`).
        if extents[last] <= 2 ** 15:
            raise NotApplicable(
                f"M1.19: {case.category!r} innermost extent {extents[last]} "
                f"does not exceed the i16 carrier span"
            )
        return Structural(kind="narrow-carrier", axis=last, tile=16)
    if mut.spec == 20:
        # Symbolic view offset (family M1-g). The emitter builds a
        # `memref.subview` whose axis-0 offset is the outermost induction
        # variable, i.e. a runtime value no static check can resolve, and reads
        # through it at the loop's own indices. The doubled coordinate overruns
        # the parent for all but the first row. The verifier cannot resolve it,
        # and RTV *does* dynamically verify the subview, so the defect is silent
        # under RTV-off and caught under RTV-on. The read index itself is
        # untouched.
        return Structural(kind="subview-symbolic", axis=last)
    raise KeyError(f"unknown M1 spec {mut.spec}")
