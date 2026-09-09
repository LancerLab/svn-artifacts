#!/usr/bin/env python3
"""Memref/affine-level emitter for the `mlir-low` lane (mutation class M1).

Why this module is separate from `emit.py`
------------------------------------------
`emit.py` composes at the tensor/linalg level, which is the `mlir-linalg` lane's
surface and the only place M2 (shape-compatibility) is expressible -- a rank or
extent mismatch is a *type* error there. M1 (element-access) is the opposite: it
is about the index actually used to reach memory, which the linalg surface hides
behind `indexing_maps`. Per `mutation-specs.md` §4 the MLIR-low translation of M1
is "`affine`/`memref` index out of range", so these kernels are built from
`memref.alloc` + `affine.for` + `affine.load`/`affine.store` and construct their
own access expressions. That is what makes an off-by-one or a negative index
representable at all.

The clean kernel is deliberately TILED, with a real boundary guard
----------------------------------------------------------------
M1 spec 1 is "dropped boundary mask (omit the `mask`/guard on a `p#n` chunked
access)". That defect is only expressible if the clean kernel *has* a guard to
drop -- otherwise M1.1 degenerates into M1.2 (off-by-one) and the two specs stop
being distinct, which §1 forbids.

So the innermost loop runs over `ceildiv(extent, TILE) * TILE` iterations, i.e. it
rounds the iteration space up to a whole number of tiles, and a guard makes the
ragged tail safe. `_tile_for` picks a TILE that does *not* divide the extent,
which is what makes the guard load-bearing rather than vacuously true. Dropping
the guard then genuinely reads and writes past the end.

Two LLVM-21 syntax traps this module encodes
--------------------------------------------
* Affine-map expressions use **infix** operators: `d0 ceildiv 2`, never
  `ceildiv(d0, 2)`. The function-call spelling fails with "use of undeclared
  identifier" on the bare `d0`, because MLIR never recognized the call and fell
  back to resolving `d0` as an SSA name.
* A guard inside a reduction must produce its value through `scf.if ... -> (f32)`
  with an `else` arm yielding the incoming accumulator. Defining the contribution
  inside a result-less `scf.if` and then `arith.select`-ing it outside is a
  use-outside-defining-region, since `scf.if` introduces a new scope.

Loc discipline (same as `emit.py`)
----------------------------------
Every op is tagged `loc("kernel")` or `loc("oracle")`. RTV bakes the location into
each assert message, and that is the only surviving attribution channel -- S9
(remainder = Σ unconditional_guards) must count kernel guards only, never the
guards RTV places on this harness's own checksum oracle.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, Sequence

import numpy as np

import compose as C
from emit import (
    Emitter,
    _flat_index,
    _position_weight,
    concrete,
    f32_lit,
)
from mutate import Structural

# Categories where M1 is measured (mutation-specs.md §5 minimal set).
LOW_CATS: list[str] = ["relu", "transpose", "softmax", "layer_normalization"]

# -inf as an f32 bit pattern: the identity for a row-max reduction.
_NEG_INF = "0xFF800000"
# LayerNorm epsilon. Spelled with a decimal point because MLIR's lexer rejects
# `1e-5` (see `emit.f32_lit`).
_EPS = "1.0e-5"


class NotExpressible(Exception):
    """The low-level surface cannot realize this defect on this category.

    Recorded as `outcome=n/a` (mutation-specs.md §6), never re-balanced and never
    counted as a detection.
    """


def guarded_needed(st: Structural | None) -> bool:
    """The guard is present unless the defect is precisely its removal."""
    return st is None or st.kind != "drop-mask"


# --------------------------------------------------------------------------
# Allocation, extents, index arithmetic
# --------------------------------------------------------------------------


def _alloc(
    e: Emitter, dims: tuple[int | None, ...], resolved: Sequence[int], name: str
) -> tuple[str, str]:
    """`memref.alloc`, supplying one index operand per dynamic extent.

    Manifest §5.2 rule 3: dynamic extents are bound to concrete small constants
    at the entry, so each `?` becomes an `arith.constant`.
    """
    mtype = C.memref_type(dims)
    args = []
    for k, d in enumerate(dims):
        if d is None:
            v = e.new("d")
            e.emit(f"{v} = arith.constant {resolved[k]} : index")
            args.append(v)
    ssa = e.new(name)
    if args:
        e.emit(f"{ssa} = memref.alloc({', '.join(args)}) : {mtype}")
    else:
        e.emit(f"{ssa} = memref.alloc() : {mtype}")
    return ssa, mtype


def _dim(e: Emitter, buf: str, mtype: str, k: int) -> str:
    """Runtime extent of axis `k`.

    Extents are read back off the memref rather than baked in as literals, for
    both static and dynamic shapes. That keeps one code path and matches how a
    real kernel receives its shape, so a mutation cannot accidentally be masked
    by a literal that disagrees with the allocation.
    """
    c = e.new("c")
    e.emit(f"{c} = arith.constant {k} : index")
    d = e.new("ext")
    e.emit(f"{d} = memref.dim {buf}, {c} : {mtype}")
    return d


def _tile_for(extent: int) -> int:
    """Pick a tile size that does NOT divide `extent`.

    The guard is load-bearing only when the rounded-up iteration space is strictly
    larger than the extent, i.e. when `extent % TILE != 0`. A tile that divides the
    extent would make the guard vacuously true and turn M1.1 into a no-op, which
    specs §7.1 discards.
    """
    if extent < 1:
        raise ValueError(f"extent {extent} cannot be tiled")
    return 2 if extent % 2 else 3


def _round_up(e: Emitter, extent: str, tile: int) -> str:
    """Round `extent` up to a whole number of tiles (infix `ceildiv`)."""
    b = e.new("b")
    e.emit(
        f"{b} = affine.apply affine_map<(d0) -> (d0 ceildiv {tile} * {tile})>"
        f"({extent})"
    )
    return b


def _apply(e: Emitter, expr: str, v: str) -> str:
    r = e.new("ix")
    e.emit(f"{r} = affine.apply affine_map<(d0) -> ({expr})>({v})")
    return r


def _perturb(e: Emitter, idx: list[str], st: Structural | None, axis: int) -> list[str]:
    """Apply an M1 defect to a read-index list. MUST be called inside the nest.

    The perturbation lands on the **read** index only; the write stays at the
    loop's own indices. That isolates the defect in the access expression, which
    is what §1 describes, instead of also corrupting the destination.

    The six kinds realize §1's six specs:
      * `drop-mask`         -- omit the guard, keep the rounded-up bound (spec 1).
                               Changes the loop structure, not an index, so it is
                               handled by `guarded_needed` at the nest level.
      * `off-by-one`        -- `j + 1` (spec 2)
      * `negative-index`    -- `j - 1`, so the first iteration reads -1 (spec 3)
      * `transposed-stride` -- swap the last two read indices (spec 4)
      * `offset-overrun`    -- `j + extent`, wholly past the end (spec 5)
      * `zero-stride`       -- `j * 0`, in range but every iteration reads element
                               0 (spec 6, the iteration-validity residue)

    Calling this before the loop is opened emits `affine.apply` referencing an
    induction variable that does not exist yet -- a use-before-def the verifier
    rejects.
    """
    if st is None or st.kind == "drop-mask":
        return idx
    k = st.axis
    if st.kind == "off-by-one":
        idx[k] = _apply(e, "d0 + 1", idx[k])
    elif st.kind == "negative-index":
        idx[k] = _apply(e, "d0 - 1", idx[k])
    elif st.kind == "offset-overrun":
        idx[k] = _apply(e, f"d0 + {st.offset}", idx[k])
    elif st.kind == "zero-stride":
        idx[k] = _apply(e, "d0 * 0", idx[k])
    elif st.kind == "transposed-stride":
        if k < 1:
            raise NotExpressible("transposed-stride needs at least two axes to swap")
        idx[k], idx[k - 1] = idx[k - 1], idx[k]
    else:
        raise KeyError(f"unhandled M1 structural kind {st.kind!r}")
    return idx


# --------------------------------------------------------------------------
# Loop nests
# --------------------------------------------------------------------------


@contextmanager
def _loops(e: Emitter, ivs: Sequence[str], bounds: Sequence[str]) -> Iterator[None]:
    """Open a perfectly nested `affine.for` over the given bounds."""
    for iv, b in zip(ivs, bounds):
        e.emit(f"affine.for {iv} = 0 to {b} {{")
        e.indent += 1
    try:
        yield
    finally:
        for _ in bounds:
            e.indent -= 1
            e.emit("}")


@contextmanager
def _tiled(
    e: Emitter,
    ivs: Sequence[str],
    bounds: Sequence[str],
    ext: str,
    st: Structural | None,
    axis: int,
) -> Iterator[list[str]]:
    """Open the tiled nest and yield the (possibly perturbed) read indices.

    The guard is emitted unless the defect *is* the guard's removal (M1.1). The
    index perturbation is applied here, inside the nest, so the `affine.apply`
    sees a bound induction variable.
    """
    guarded = guarded_needed(st)
    for iv, b in zip(ivs, bounds):
        e.emit(f"affine.for {iv} = 0 to {b} {{")
        e.indent += 1
    if guarded:
        inb = e.new("inb")
        e.emit(f"{inb} = arith.cmpi slt, {ivs[axis]}, {ext} : index")
        e.emit(f"scf.if {inb} {{")
        e.indent += 1
    try:
        yield _perturb(e, list(ivs), st, axis)
    finally:
        if guarded:
            e.indent -= 1
            e.emit("}")
        for _ in bounds:
            e.indent -= 1
            e.emit("}")


def _guarded_reduce(
    e: Emitter,
    outer: Sequence[str],
    buf: str,
    buf_t: str,
    ext: str,
    bound: str,
    axis: int,
    init: str,
    st: Structural | None,
    body: Callable[[Emitter, list[str], str, str], str],
) -> str:
    """One guarded reduction over the trailing axis; returns the result SSA.

    `body(e, read_indices, acc, iv)` emits the per-iteration contribution and
    returns its SSA name. `iv` is the loop's own induction variable, supplied so a
    body that also *writes* can target the unperturbed index: the defect belongs
    in the read expression alone (§1), and letting it leak into the write would
    corrupt the destination too and blur which access was actually out of range.

    The body runs inside the guard when there is one, so a dropped guard (M1.1)
    makes it execute for the whole rounded-up range -- which is exactly the
    out-of-bounds access the spec describes.

    The guard threads its value out through `scf.if ... -> (f32)` with an `else`
    arm yielding the incoming accumulator. The alternative -- defining the
    contribution inside a result-less `scf.if` and `arith.select`-ing it
    afterwards -- is a use-outside-defining-region, because `scf.if` opens a new
    scope.
    """
    guarded = guarded_needed(st)
    j = e.new("j")
    acc = e.new("a")
    res = e.new("r")
    e.emit(
        f"{res} = affine.for {j} = 0 to {bound} "
        f"iter_args({acc} = {init}) -> (f32) {{"
    )
    e.indent += 1

    inb = None
    if guarded:
        inb = e.new("inb")
        e.emit(f"{inb} = arith.cmpi slt, {j}, {ext} : index")
        # The `scf.if` carries a result type, so the guarded contribution can be
        # threaded out. `contrib` itself is scoped inside the `if` region and
        # cannot be referenced by the `affine.yield` below.
        sel = e.new("g")
        e.emit(f"{sel} = scf.if {inb} -> (f32) {{")
        e.indent += 1

    ridx = _perturb(e, list(outer) + [j], st, axis)
    contrib = body(e, ridx, acc, j)

    if guarded:
        e.emit(f"scf.yield {contrib} : f32")
        e.indent -= 1
        e.emit("} else {")
        e.indent += 1
        e.emit(f"scf.yield {acc} : f32")
        e.indent -= 1
        e.emit("}")
        e.emit(f"affine.yield {sel} : f32")
    else:
        e.emit(f"affine.yield {contrib} : f32")

    e.indent -= 1
    e.emit("}")
    return res


# --------------------------------------------------------------------------
# Input fill (same deterministic formula as the tensor-level lane)
# --------------------------------------------------------------------------


def _fill_formula(e: Emitter, buf: str, mtype: str, nd: int, offset: int = 0) -> None:
    """Fill `buf` with `((flat_index + offset) * 37) % 13 - 6`, matching
    `compose.input_values` exactly so the numpy oracle agrees bit-for-bit.

    Not optional. A bare `memref.alloc` is uninitialized, and an unfilled input
    makes even the *clean* kernel report `corrupts` -- which silently invalidates
    every verdict measured against it.
    """
    from emit import _flat_index_body

    ivs = [e.new("i") for _ in range(nd)]
    bounds = [_dim(e, buf, mtype, k) for k in range(nd)]
    with _loops(e, ivs, bounds):
        val = _flat_index_body(e, ivs, bounds, offset)
        e.emit(f"affine.store {val}, {buf}[{', '.join(ivs)}] : {mtype}")


def _zero_fill(e: Emitter, buf: str, mtype: str, nd: int) -> None:
    """Zero the output buffer.

    Not cosmetic: a tile the kernel fails to write must read back as a
    deterministic 0 rather than whatever the allocator handed back, or the
    oracle's verdict on a partial-write defect becomes run-dependent.
    """
    z = e.new("c")
    e.emit(f"{z} = arith.constant 0.0 : f32")
    ivs = [e.new("z") for _ in range(nd)]
    bounds = [_dim(e, buf, mtype, k) for k in range(nd)]
    with _loops(e, ivs, bounds):
        e.emit(f"affine.store {z}, {buf}[{', '.join(ivs)}] : {mtype}")


# --------------------------------------------------------------------------
# Oracle: full reduction over a memref, then compare against baked checksums
# --------------------------------------------------------------------------


def _reduce_low(e: Emitter, buf: str, mtype: str, nd: int, square: bool = False,
                weighted: bool = False) -> str:
    """Reduce `buf` to one f32 via `affine.for` `iter_args`, threading a single
    accumulator through the nest.

    Three modes, matching the three checksums in `compose.checksums`:
    plain sum, sum of squares (`square`), or position-weighted sum (`weighted`).
    The weighted mode is what makes the oracle order-sensitive; without it an M1
    defect that merely *reorders* values -- off-by-one, negative index,
    transposed stride -- leaves both permutation-invariant checksums untouched and
    the mutant is scored `noop`, a false success specs §7.1 requires be
    discarded.

    The weight is derived from the induction variables and the buffer's *true*
    extents (`_dim`), never from literals, so a mutant cannot have its position
    arithmetic silently agree with the clean kernel's.

    Unguarded on purpose: it walks the true extent, and RTV's bounds checks on
    these loads are attributed to `loc("oracle")` so they stay out of S9.
    """
    zero = e.new("c")
    e.emit(f"{zero} = arith.constant 0.0 : f32")
    ivs = [e.new("i") for _ in range(nd)]
    bounds = [_dim(e, buf, mtype, k) for k in range(nd)]
    accs = [e.new("a") for _ in range(nd)]
    res = [e.new("r") for _ in range(nd)]

    for k in range(nd):
        init = zero if k == 0 else accs[k - 1]
        e.emit(
            f"{res[k]} = affine.for {ivs[k]} = 0 to {bounds[k]} "
            f"iter_args({accs[k]} = {init}) -> (f32) {{"
        )
        e.indent += 1

    v = e.new("x")
    e.emit(f"{v} = affine.load {buf}[{', '.join(ivs)}] : {mtype}")
    if square:
        sq = e.new("t")
        e.emit(f"{sq} = arith.mulf {v}, {v} : f32")
        term = sq
    elif weighted:
        flat = _flat_index(e, ivs, bounds)
        w = _position_weight(e, flat)
        p = e.new("t")
        e.emit(f"{p} = arith.mulf {v}, {w} : f32")
        term = p
    else:
        term = v

    add = e.new("t")
    e.emit(f"{add} = arith.addf {accs[-1]}, {term} : f32")

    for k in reversed(range(nd)):
        val = add if k == nd - 1 else res[k + 1]
        e.emit(f"affine.yield {val} : f32")
        e.indent -= 1
        e.emit("}")
    return res[0]


def emit_oracle_low(e: Emitter, out: str, out_mtype: str, nd: int, ref: np.ndarray) -> str:
    """Compare the output's three checksums against the numpy reference; i32 flag.

    Each check gets its own tolerance via `compose.tolerance`, which is *relative*
    to that checksum's magnitude. The previous absolute budget (`1e-2*sqrt(nel)`)
    was ~1000x too loose and scored a real softmax drop-mask defect as `noop`: the
    defect moves `sum-of-squares` by 1.35e-2, under the 2.83e-2 absolute budget,
    while the clean kernel's largest relative discrepancy is only 8.2e-7.
    """
    ref_sum, ref_sq, ref_w = C.checksums(ref)
    nel = int(ref.size)
    tol_s = C.tolerance(nel, ref_sum)
    tol_q = C.tolerance(nel, ref_sq)
    tol_w = C.tolerance(nel, ref_w)

    s = _reduce_low(e, out, out_mtype, nd, square=False)
    q = _reduce_low(e, out, out_mtype, nd, square=True)
    wv = _reduce_low(e, out, out_mtype, nd, weighted=True)

    rs = e.new("c")
    e.emit(f"{rs} = arith.constant {f32_lit(ref_sum)} : f32")
    rq = e.new("c")
    e.emit(f"{rq} = arith.constant {f32_lit(ref_sq)} : f32")
    rw = e.new("c")
    e.emit(f"{rw} = arith.constant {f32_lit(ref_w)} : f32")
    ct = e.new("c")
    e.emit(f"{ct} = arith.constant {f32_lit(tol_s)} : f32")
    cq = e.new("c")
    e.emit(f"{cq} = arith.constant {f32_lit(tol_q)} : f32")
    cw = e.new("c")
    e.emit(f"{cw} = arith.constant {f32_lit(tol_w)} : f32")

    ds = e.new("t")
    e.emit(f"{ds} = arith.subf {s}, {rs} : f32")
    ad = e.new("t")
    e.emit(f"{ad} = math.absf {ds} : f32")
    bad1 = e.new("t")
    e.emit(f'{bad1} = arith.cmpf "ogt", {ad}, {ct} : f32')

    dq = e.new("t")
    e.emit(f"{dq} = arith.subf {q}, {rq} : f32")
    aq = e.new("t")
    e.emit(f"{aq} = math.absf {dq} : f32")
    bad2 = e.new("t")
    e.emit(f'{bad2} = arith.cmpf "ogt", {aq}, {cq} : f32')

    dw = e.new("t")
    e.emit(f"{dw} = arith.subf {wv}, {rw} : f32")
    aw = e.new("t")
    e.emit(f"{aw} = math.absf {dw} : f32")
    bad3 = e.new("t")
    e.emit(f'{bad3} = arith.cmpf "ogt", {aw}, {cw} : f32')

    bad12 = e.new("t")
    e.emit(f"{bad12} = arith.ori {bad1}, {bad2} : i1")
    bad = e.new("t")
    e.emit(f"{bad} = arith.ori {bad12}, {bad3} : i1")
    c0 = e.new("c")
    e.emit(f"{c0} = arith.constant 0 : i32")
    c1 = e.new("c")
    e.emit(f"{c1} = arith.constant 1 : i32")
    r = e.new("r")
    e.emit(f"{r} = arith.select {bad}, {c1}, {c0} : i32")
    return r


# --------------------------------------------------------------------------
# Operator bodies (memref/affine surface)
# --------------------------------------------------------------------------


def _low_relu(e: Emitter, case: C.Case, st: Structural | None):
    inp_dims, out_dims = case.dims["inp"], case.dims["out"]
    inp_r = concrete(inp_dims, case.dyn, "inp")
    out_r = concrete(out_dims, case.dyn, "out")
    nd = len(inp_dims)
    axis = nd - 1

    inp, inp_t = _alloc(e, inp_dims, inp_r, "inp")
    _fill_formula(e, inp, inp_t, nd, offset=0)
    out, out_t = _alloc(e, out_dims, out_r, "out")
    _zero_fill(e, out, out_t, nd)

    ext = _dim(e, inp, inp_t, axis)
    bound = _round_up(e, ext, _tile_for(inp_r[axis]))
    ivs = [e.new("i") for _ in range(nd)]
    bounds = [_dim(e, inp, inp_t, k) for k in range(axis)] + [bound]

    z = e.new("c")
    e.emit(f"{z} = arith.constant 0.0 : f32")
    with _tiled(e, ivs, bounds, ext, st, axis) as ridx:
        v = e.new("v")
        e.emit(f"{v} = affine.load {inp}[{', '.join(ridx)}] : {inp_t}")
        # `arith.maxf` was renamed in LLVM 21; `maximumf` propagates NaN like
        # numpy.maximum, which is what the reference oracle uses.
        r = e.new("r")
        e.emit(f"{r} = arith.maximumf {v}, {z} : f32")
        e.emit(f"affine.store {r}, {out}[{', '.join(ivs)}] : {out_t}")
    return out, out_t, out_r


def _low_transpose(e: Emitter, case: C.Case, st: Structural | None):
    inp_dims, out_dims = case.dims["inp"], case.dims["out"]
    inp_r = concrete(inp_dims, case.dyn, "inp")
    out_r = concrete(out_dims, case.dyn, "out")
    nd = len(inp_dims)
    axis = nd - 1

    inp, inp_t = _alloc(e, inp_dims, inp_r, "inp")
    _fill_formula(e, inp, inp_t, nd, offset=0)
    out, out_t = _alloc(e, out_dims, out_r, "out")
    _zero_fill(e, out, out_t, nd)

    ext = _dim(e, inp, inp_t, axis)
    bound = _round_up(e, ext, _tile_for(inp_r[axis]))
    ivs = [e.new("i") for _ in range(nd)]
    bounds = [_dim(e, inp, inp_t, k) for k in range(axis)] + [bound]

    with _tiled(e, ivs, bounds, ext, st, axis) as ridx:
        v = e.new("v")
        e.emit(f"{v} = affine.load {inp}[{', '.join(ridx)}] : {inp_t}")
        widx = list(reversed(ivs))
        e.emit(f"affine.store {v}, {out}[{', '.join(widx)}] : {out_t}")
    return out, out_t, out_r


def _low_softmax(e: Emitter, case: C.Case, st: Structural | None):
    """Row-wise softmax over the trailing axis, rank-generic.

    Three passes per row (max, exp+sum, normalize), the standard formulation and
    the one `compose.reference` implements.
    """
    inp_dims, out_dims = case.dims["inp"], case.dims["out"]
    inp_r = concrete(inp_dims, case.dyn, "inp")
    out_r = concrete(out_dims, case.dyn, "out")
    nd = len(inp_dims)
    axis = nd - 1

    inp, inp_t = _alloc(e, inp_dims, inp_r, "inp")
    _fill_formula(e, inp, inp_t, nd, offset=0)
    out, out_t = _alloc(e, out_dims, out_r, "out")
    _zero_fill(e, out, out_t, nd)

    ext = _dim(e, inp, inp_t, axis)
    bound = _round_up(e, ext, _tile_for(inp_r[axis]))
    outer = [e.new("p") for _ in range(axis)]
    obounds = [_dim(e, inp, inp_t, k) for k in range(axis)]

    ninf = e.new("c")
    e.emit(f"{ninf} = arith.constant {_NEG_INF} : f32")
    zero = e.new("c")
    e.emit(f"{zero} = arith.constant 0.0 : f32")

    with _loops(e, outer, obounds):
        # --- pass 1: row max -------------------------------------------------
        def _maxbody(e: Emitter, ridx: list[str], acc: str, iv: str) -> str:
            v = e.new("v")
            e.emit(f"{v} = affine.load {inp}[{', '.join(ridx)}] : {inp_t}")
            m = e.new("t")
            e.emit(f"{m} = arith.maximumf {acc}, {v} : f32")
            return m

        mx = _guarded_reduce(
            e, outer, inp, inp_t, ext, bound, axis, ninf, st, _maxbody
        )

        # --- pass 2: exp and row sum -----------------------------------------
        def _expbody(e: Emitter, ridx: list[str], acc: str, iv: str) -> str:
            v = e.new("v")
            e.emit(f"{v} = affine.load {inp}[{', '.join(ridx)}] : {inp_t}")
            d = e.new("t")
            e.emit(f"{d} = arith.subf {v}, {mx} : f32")
            ex = e.new("t")
            e.emit(f"{ex} = math.exp {d} : f32")
            # Written at the loop's own index, not the perturbed read index, so
            # the defect stays isolated in the read. The store is still inside the
            # guard, so dropping the guard (M1.1) overruns the destination too.
            widx = list(outer) + [iv]
            e.emit(f"affine.store {ex}, {out}[{', '.join(widx)}] : {out_t}")
            ns = e.new("t")
            e.emit(f"{ns} = arith.addf {acc}, {ex} : f32")
            return ns

        sm = _guarded_reduce(
            e, outer, inp, inp_t, ext, bound, axis, zero, st, _expbody
        )

        # --- pass 3: normalize (unguarded; walks the true extent) ------------
        k = e.new("k")
        e.emit(f"affine.for {k} = 0 to {ext} {{")
        e.indent += 1
        widx3 = list(outer) + [k]
        ev = e.new("v")
        e.emit(f"{ev} = affine.load {out}[{', '.join(widx3)}] : {out_t}")
        nv = e.new("t")
        e.emit(f"{nv} = arith.divf {ev}, {sm} : f32")
        e.emit(f"affine.store {nv}, {out}[{', '.join(widx3)}] : {out_t}")
        e.indent -= 1
        e.emit("}")

    return out, out_t, out_r


def _low_layer_norm(e: Emitter, case: C.Case, st: Structural | None):
    """Layer normalization over the trailing axis, rank-generic."""
    lhs_dims = case.dims["lhs"]
    out_dims = case.dims["out"]
    sc_dims = case.dims["scale"]
    bi_dims = case.dims["bias"]
    lhs_r = concrete(lhs_dims, case.dyn, "lhs")
    out_r = concrete(out_dims, case.dyn, "out")
    sc_r = concrete(sc_dims, case.dyn, "scale")
    bi_r = concrete(bi_dims, case.dyn, "bias")
    nd = len(lhs_dims)
    axis = nd - 1

    lhs, lhs_t = _alloc(e, lhs_dims, lhs_r, "lhs")
    _fill_formula(e, lhs, lhs_t, nd, offset=0)
    scale, scale_t = _alloc(e, sc_dims, sc_r, "scale")
    _fill_formula(e, scale, scale_t, len(sc_dims), offset=2000)
    bias, bias_t = _alloc(e, bi_dims, bi_r, "bias")
    _fill_formula(e, bias, bias_t, len(bi_dims), offset=3000)
    out, out_t = _alloc(e, out_dims, out_r, "out")
    _zero_fill(e, out, out_t, nd)

    ext = _dim(e, lhs, lhs_t, axis)
    bound = _round_up(e, ext, _tile_for(lhs_r[axis]))
    outer = [e.new("p") for _ in range(axis)]
    obounds = [_dim(e, lhs, lhs_t, k) for k in range(axis)]

    zero = e.new("c")
    e.emit(f"{zero} = arith.constant 0.0 : f32")

    with _loops(e, outer, obounds):
        # --- mean ------------------------------------------------------------
        def _sumbody(e: Emitter, ridx: list[str], acc: str, iv: str) -> str:
            v = e.new("v")
            e.emit(f"{v} = affine.load {lhs}[{', '.join(ridx)}] : {lhs_t}")
            a = e.new("t")
            e.emit(f"{a} = arith.addf {acc}, {v} : f32")
            return a

        tot = _guarded_reduce(
            e, outer, lhs, lhs_t, ext, bound, axis, zero, st, _sumbody
        )

        nf = e.new("t")
        e.emit(f"{nf} = arith.index_cast {ext} : index to i64")
        nf32 = e.new("t")
        e.emit(f"{nf32} = arith.sitofp {nf} : i64 to f32")
        mean = e.new("mean")
        e.emit(f"{mean} = arith.divf {tot}, {nf32} : f32")

        # --- variance --------------------------------------------------------
        def _sqbody(e: Emitter, ridx: list[str], acc: str, iv: str) -> str:
            v = e.new("v")
            e.emit(f"{v} = affine.load {lhs}[{', '.join(ridx)}] : {lhs_t}")
            d = e.new("t")
            e.emit(f"{d} = arith.subf {v}, {mean} : f32")
            m = e.new("t")
            e.emit(f"{m} = arith.mulf {d}, {d} : f32")
            a = e.new("t")
            e.emit(f"{a} = arith.addf {acc}, {m} : f32")
            return a

        sq = _guarded_reduce(
            e, outer, lhs, lhs_t, ext, bound, axis, zero, st, _sqbody
        )

        var = e.new("var")
        e.emit(f"{var} = arith.divf {sq}, {nf32} : f32")
        eps = e.new("c")
        e.emit(f"{eps} = arith.constant {_EPS} : f32")
        ve = e.new("t")
        e.emit(f"{ve} = arith.addf {var}, {eps} : f32")
        sd = e.new("sd")
        e.emit(f"{sd} = math.sqrt {ve} : f32")

        # --- normalize + affine (unguarded; walks the true extent) -----------
        k = e.new("k")
        e.emit(f"affine.for {k} = 0 to {ext} {{")
        e.indent += 1
        ridx_k = _perturb(e, list(outer) + [k], st, axis)
        xv = e.new("v")
        e.emit(f"{xv} = affine.load {lhs}[{', '.join(ridx_k)}] : {lhs_t}")
        d2 = e.new("t")
        e.emit(f"{d2} = arith.subf {xv}, {mean} : f32")
        n2 = e.new("t")
        e.emit(f"{n2} = arith.divf {d2}, {sd} : f32")
        sc = e.new("t")
        e.emit(f"{sc} = affine.load {scale}[{k}] : {scale_t}")
        m1 = e.new("t")
        e.emit(f"{m1} = arith.mulf {n2}, {sc} : f32")
        bi = e.new("t")
        e.emit(f"{bi} = affine.load {bias}[{k}] : {bias_t}")
        m2 = e.new("t")
        e.emit(f"{m2} = arith.addf {m1}, {bi} : f32")
        widx = list(outer) + [k]
        e.emit(f"affine.store {m2}, {out}[{', '.join(widx)}] : {out_t}")
        e.indent -= 1
        e.emit("}")

    return out, out_t, out_r


LOW_EMITTERS = {
    "relu": _low_relu,
    "transpose": _low_transpose,
    "softmax": _low_softmax,
    "layer_normalization": _low_layer_norm,
}


# --------------------------------------------------------------------------
# Top-level module assembly
# --------------------------------------------------------------------------


def emit_kernel_low(
    case: C.Case,
    ref_case: C.Case | None = None,
    structural: Structural | None = None,
) -> str:
    """Compose a complete memref-level module for a (possibly mutated) case.

    `ref_case` is the *unmutated* case whose reference output is baked into the
    oracle. A mutant must always be passed the clean case: comparing a mutant
    against its own mutated reference would make every mutant look correct.
    """
    if case.category not in LOW_EMITTERS:
        raise KeyError(f"no low-level emitter for category {case.category!r}")
    if ref_case is None:
        ref_case = case

    e = Emitter()
    e.indent = 0
    e.raw("func.func @main() -> i32 {")
    e.indent += 1

    with e.location("kernel"):
        out, out_t, out_r = LOW_EMITTERS[case.category](e, case, structural)

    nd = len(out_r)
    ref = C.reference(ref_case)
    with e.location("oracle"):
        res = emit_oracle_low(e, out, out_t, nd, ref)
    e.emit(f"return {res} : i32")
    e.indent -= 1
    e.raw("}")
    return e.text()
