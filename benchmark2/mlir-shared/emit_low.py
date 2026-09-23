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

# Threads per block. The trailing axis is strided over by this many lanes in a
# single block; the axes outside it are one block each. 256 is the smallest
# multiple of the 32-lane warp that still lets the carrier extent (40000) be
# covered in a bounded number of strides without exceeding the shared-memory the
# `gpu.all_reduce` lowering stages per block.
_GPU_BLOCK = 256


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


def _view_type(nd: int) -> str:
    """Result type of `memref.subview` with all-dynamic size and stride.

    Every size and stride operand is an SSA value, so the result layout must be
    dynamic on every axis; a partially static layout (`strided<[?, 1], ...>`)
    fails the subview verifier with "mismatch of result layout".
    """
    dims = "x".join("?" for _ in range(nd))
    strides = ", ".join("?" for _ in range(nd))
    return f"memref<{dims}xf32, strided<[{strides}], offset: ?>>"


def _subview(e: Emitter, buf: str, buf_t: str, dims: Sequence[int | None],
             off0: str) -> tuple[str, str]:
    """A rank-preserving `memref.subview` of `buf` offset by `off0` on axis 0.

    `off0` is a live SSA value (the outermost induction variable), which is what
    keeps the descriptor symbolic: a statically resolvable offset is refused, but
    this one is neither refused nor checked. The size operands are the parent's
    true extents and every stride is 1, so the subview's own shape is satisfied
    and the overrun only appears in the parent address the reads actually reach.
    """
    nd = len(dims)
    c0 = e.new("c")
    e.emit(f"{c0} = arith.constant 0 : index")
    c1 = e.new("c")
    e.emit(f"{c1} = arith.constant 1 : index")
    offsets = [off0] + [c0] * (nd - 1)
    sizes = [_dim(e, buf, buf_t, k) for k in range(nd)]
    vt = _view_type(nd)
    sv = e.new("sv")
    e.emit(
        f"{sv} = memref.subview {buf}[{', '.join(offsets)}] "
        f"[{', '.join(sizes)}] [{', '.join([c1] * nd)}] : {buf_t} to {vt}"
    )
    return sv, vt


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


def _cidx(e: Emitter, n: int) -> str:
    """An `index` constant. Used inside GPU kernels, where `affine.apply` is not
    available (the kernel region is lowered by the NVVM pipeline, which runs
    `scf-to-cf` before `lower-affine`), so every index adjustment the M1
    perturbations need is spelled with `arith`."""
    c = e.new("c")
    e.emit(f"{c} = arith.constant {n} : index")
    return c


def _addi(e: Emitter, a: str, b: str) -> str:
    r = e.new("ix")
    e.emit(f"{r} = arith.addi {a}, {b} : index")
    return r


def _subi(e: Emitter, a: str, b: str) -> str:
    r = e.new("ix")
    e.emit(f"{r} = arith.subi {a}, {b} : index")
    return r


def _muli(e: Emitter, a: str, b: str) -> str:
    r = e.new("ix")
    e.emit(f"{r} = arith.muli {a}, {b} : index")
    return r


def _remsi(e: Emitter, a: str, b: str) -> str:
    r = e.new("ix")
    e.emit(f"{r} = arith.remsi {a}, {b} : index")
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

    The v2.1 additions the memref surface can realise:
      * `broadcast-index`   -- reuse a different induction variable for this axis
                               (spec 12, family M1-d). In range at small size, so
                               the miss is silent rather than an RTV catch.
      * `tile-coord`        -- advance the index by a whole tile (spec 14, family
                               M1-e); the in-tile offset is intact, the tile
                               coordinate is one past the tiled extent.
      * `overlap-write`     -- not a read perturbation; `_write_perturb` wraps the
                               store index instead (spec 11, family M1-h). Ignored
                               here so the read stays clean.
      * `subview-symbolic`  -- not an index perturbation either; the caller
                               replaces the read buffer with a `memref.subview`
                               whose offset is a runtime value (spec 20, family
                               M1-g). Ignored here so the read index stays clean.

    Calling this before the loop is opened emits `affine.apply` referencing an
    induction variable that does not exist yet -- a use-before-def the verifier
    rejects.
    """
    if (st is None or st.kind.startswith("m4-")
            or st.kind in ("drop-mask", "overlap-write", "subview-symbolic")):
        # M4 is a loop-bound class, not an index perturbation: `_m4_triple`
        # rewrites the bound at the loop site and the read index stays clean.
        return idx
    k = st.axis
    if st.kind == "off-by-one":
        idx[k] = _addi(e, idx[k], _cidx(e, 1))
    elif st.kind == "negative-index":
        idx[k] = _subi(e, idx[k], _cidx(e, 1))
    elif st.kind == "offset-overrun":
        idx[k] = _addi(e, idx[k], _cidx(e, st.offset))
    elif st.kind == "zero-stride":
        idx[k] = _muli(e, idx[k], _cidx(e, 0))
    elif st.kind == "transposed-stride":
        if k < 1:
            raise NotExpressible("transposed-stride needs at least two axes to swap")
        idx[k], idx[k - 1] = idx[k - 1], idx[k]
    elif st.kind == "broadcast-index":
        # Reuse the outermost induction variable for this axis (M1.12). The
        # wrong variable is in range at the small battery size, which is what
        # makes the defect silent: no bounds check fires.
        if k < 1:
            raise NotExpressible("broadcast-index needs a rank >= 2 to reuse an axis")
        idx[k] = idx[0]
    elif st.kind == "tile-coord":
        # Advance by one whole tile (M1.14). `st.tile` is the axis extent, so the
        # tile size comes from the same rule the clean nest uses.
        idx[k] = _addi(e, idx[k], _cidx(e, _tile_for(st.tile)))
    elif st.kind == "narrow-carrier":
        # Carry the element index in an integer narrower than the axis (M1.19).
        # `st.tile` is the carrier bit width. The clean kernel carries the index
        # in `index`; the mutant round-trips it through `i16`, so a coordinate the
        # allocation admits (>= 2**(tile-1)) wraps to a negative offset. The read
        # must then go through `memref.load`, whose index operand may be an
        # arbitrary SSA value -- `affine.load` rejects a non-dimension-id operand
        # at `--lower-affine` (probe: "operand cannot be used as a dimension id").
        #
        # The round-trip is spelled `index -> i64 -> i16 -> index`, not the direct
        # `index_cast index -> i16 -> index`: the RTV-on pipeline runs
        # `canonicalize`, which folds the direct form straight back to the source
        # value (source and result types are both `index`), silently turning the
        # mutant into the clean kernel. The `trunci` through `i64` has no such
        # fold, so the carrier survives both pipelines. See mlir-shared/README.md.
        bits = st.tile
        w = e.new("nrw")
        e.emit(f"{w} = arith.index_cast {idx[k]} : index to i64")
        t = e.new("nrc")
        e.emit(f"{t} = arith.trunci {w} : i64 to i{bits}")
        r = e.new("ix")
        e.emit(f"{r} = arith.index_cast {t} : i{bits} to index")
        idx[k] = r
    else:
        raise KeyError(f"unhandled M1 structural kind {st.kind!r}")
    return idx


def _write_perturb(
    e: Emitter, widx: list[str], st: Structural | None, axis: int
) -> list[str]:
    """Apply an M1 defect that lives in the **store** index, not the read.

    Only `overlap-write` (spec 11, family M1-h) is a write-side defect: the store
    index is wrapped into a region `st.tile` slots wide, so several concurrent
    loop iterations address the same live slot and the last one wins. Every other
    kind leaves the store untouched (the defect belongs in the read expression).

    `axis` is supplied by the caller because it indexes *this* store's coordinate
    list, which is not always the read list's `st.axis` (transpose reverses the
    store coordinates).
    """
    if st is None or st.kind != "overlap-write":
        return widx
    widx[axis] = _remsi(e, widx[axis], _cidx(e, st.tile))
    return widx


def _read(
    e: Emitter, buf: str, buf_t: str, idx: Sequence[str], st: Structural | None
) -> str:
    """Load `buf[idx]`.

    Always `memref.load`. The GPU surface derives every index from a thread/block
    coordinate (or from `arith` arithmetic on one), which is not an affine
    dimension id, so `affine.load` -- legal only when the index is a dimension id
    -- is unavailable. `memref.load` accepts arbitrary SSA indices and lowers to
    the same `llvm.getelementptr`, and RTV instruments it identically.
    """
    v = e.new("v")
    e.emit(f"{v} = memref.load {buf}[{', '.join(idx)}] : {buf_t}")
    return v


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
def _launch(e: Emitter, grid: str, nblock: str) -> Iterator[tuple[str, str]]:
    """Open a one-dimensional `gpu.launch` and yield `(blockIdx.x, threadIdx.x)`.

    A 1-D grid indexes the iteration space *outside* the trailing axis (one
    thread-block per row/tile); a 1-D block strides over the trailing axis. That
    keeps the trailing-axis guard -- the boundary mask M1.1 drops -- a real,
    per-element predicate rather than an artifact of the grid shape, and it works
    for the carrier shapes whose trailing extent (40000) exceeds any block size.
    """
    bx, by, bz = e.new("bx"), e.new("by"), e.new("bz")
    tx, ty, tz = e.new("tx"), e.new("ty"), e.new("tz")
    c1 = _cidx(e, 1)
    gx, gy, gz = e.new("gx"), e.new("gy"), e.new("gz")
    b1, b2, b3 = e.new("b"), e.new("b"), e.new("b")
    e.emit(
        f"gpu.launch blocks({bx}, {by}, {bz}) in "
        f"({gx} = {grid}, {gy} = {c1}, {gz} = {c1}) "
        f"threads({tx}, {ty}, {tz}) in "
        f"({b1} = {nblock}, {b2} = {c1}, {b3} = {c1}) {{"
    )
    e.indent += 1
    try:
        yield bx, tx
    finally:
        e.emit("gpu.terminator")
        e.indent -= 1
        e.emit("}")


def _decompose(e: Emitter, base: str, bounds: Sequence[str]) -> list[str]:
    """Decompose `base` into row-major coordinates over `bounds`.

    `bounds` are the live extents of the axes outside the trailing one, so the
    coordinates are always in range and only the trailing axis (whose loop is
    guarded) can leave its extent.
    """
    nd = len(bounds)
    coords: list[str] = [""] * nd
    rem = base
    for k in range(nd - 1, -1, -1):
        if k == 0:
            coords[0] = rem
        else:
            c = e.new("ci")
            e.emit(f"{c} = arith.remsi {rem}, {bounds[k]} : index")
            n = e.new("cd")
            e.emit(f"{n} = arith.divsi {rem}, {bounds[k]} : index")
            coords[k] = c
            rem = n
    return coords


@contextmanager
def _guarded_axis_loop(
    e: Emitter, tx: str, bound: str, nblock: str, ext: str, st: Structural | None
) -> Iterator[str]:
    """Strided loop over the trailing axis, guarded by `j < ext`.

    The guard is the boundary mask of the chunked access: the loop runs to the
    rounded-up `bound`, and the predicate keeps the ragged tail safe. `drop-mask`
    (M1.1) is exactly the removal of this predicate, so it is emitted unless the
    defect *is* the removal.
    """
    j = e.new("j")
    lo, hi, step = _m4_triple(e, st, tx, bound, nblock, ext)
    e.emit(f"scf.for {j} = {lo} to {hi} step {step} {{")
    e.indent += 1
    guarded = guarded_needed(st)
    if guarded:
        inb = e.new("inb")
        e.emit(f"{inb} = arith.cmpi slt, {j}, {ext} : index")
        e.emit(f"scf.if {inb} {{")
        e.indent += 1
    try:
        yield j
    finally:
        if guarded:
            e.indent -= 1
            e.emit("}")
        e.indent -= 1
        e.emit("}")


def _block_reduce(
    e: Emitter,
    outer: Sequence[str],
    ext: str,
    bound: str,
    axis: int,
    init: str,
    st: Structural | None,
    tx: str,
    nblock: str,
    body: Callable[[Emitter, list[str], str, str], str],
    reduce_op: str,
) -> str:
    """Block-parallel guarded reduction over the trailing axis.

    Each thread accumulates its strided share (`body`), then `gpu.all_reduce`
    folds the per-thread partials with `reduce_op` across the whole block. The
    block's shared-memory staging is supplied by the `gpu.all_reduce` lowering.

    `body(e, read_indices, acc, iv)` mirrors `_guarded_reduce`: it runs inside the
    guard, so a dropped guard (M1.1) makes it execute over the rounded-up range.
    The read index is perturbed; a body that also writes targets the loop's own
    `iv`, so the defect stays isolated in the read.
    """
    guarded = guarded_needed(st)
    j = e.new("j")
    acc = e.new("a")
    res = e.new("r")
    lo, hi, step = _m4_triple(e, st, tx, bound, nblock, ext)
    e.emit(
        f"{res} = scf.for {j} = {lo} to {hi} step {step} "
        f"iter_args({acc} = {init}) -> (f32) {{"
    )
    e.indent += 1
    if guarded:
        inb = e.new("inb")
        e.emit(f"{inb} = arith.cmpi slt, {j}, {ext} : index")
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
        e.emit(f"scf.yield {sel} : f32")
    else:
        e.emit(f"scf.yield {contrib} : f32")
    e.indent -= 1
    e.emit("}")

    red = e.new("red")
    e.emit(f"{red} = gpu.all_reduce {res} uniform {{")
    e.indent += 1
    la, lb = e.new("la"), e.new("lb")
    e.emit(f"^bb({la} : f32, {lb} : f32):")
    e.indent += 1
    m = e.new("m")
    e.emit(f"{m} = {reduce_op} {la}, {lb} : f32")
    e.emit(f'"gpu.yield"({m}) : (f32) -> ()')
    e.indent -= 1
    e.indent -= 1
    e.emit("} : (f32) -> (f32)")
    return red


# --------------------------------------------------------------------------
# Device-buffer prologue / epilogue
# --------------------------------------------------------------------------


def _gpu_alloc(
    e: Emitter, dims: Sequence[int | None], resolved: Sequence[int],
    mtype: str, tok: str, name: str,
) -> tuple[str, str]:
    """`gpu.alloc async`, mirroring `_alloc` (dynamic extents bound to constants).

    Device memory specifically: `gpu.host_register`/`host_shared` allocations are
    invisible to compute-sanitizer's out-of-bounds checks (it tracks only CUDA
    device allocations), so they cannot carry the S12 surface.
    """
    args = []
    for k, d in enumerate(dims):
        if d is None:
            v = e.new("d")
            e.emit(f"{v} = arith.constant {resolved[k]} : index")
            args.append(v)
    ssa = e.new(name)
    t = e.new("t")
    if args:
        e.emit(f"{ssa}, {t} = gpu.alloc async [{tok}] ({', '.join(args)}) : {mtype}")
    else:
        e.emit(f"{ssa}, {t} = gpu.alloc async [{tok}] () : {mtype}")
    return ssa, t


def _gpu_memcpy(e: Emitter, tok: str, dst: str, src: str, mtype: str) -> str:
    t = e.new("t")
    e.emit(f"{t} = gpu.memcpy async [{tok}] {dst}, {src} : {mtype}, {mtype}")
    return t


def _gpu_wait(e: Emitter, tok: str) -> None:
    e.emit(f"gpu.wait [{tok}]")


def _prod(e: Emitter, vals: Sequence[str]) -> str:
    """Product of live `index` operands (for the grid size)."""
    if not vals:
        return _cidx(e, 1)
    acc = vals[0]
    for v in vals[1:]:
        acc = _muli(e, acc, v)
    return acc


# --------------------------------------------------------------------------
# Input fill (same deterministic formula as the tensor-level lane)
# --------------------------------------------------------------------------


def _fill_formula(e: Emitter, buf: str, mtype: str, nd: int, offset: int = 0) -> None:
    """Fill `buf` with `((flat_index + offset) * 37) % 13 - 6`, matching
    `compose.input_values` exactly so the numpy oracle agrees bit-for-bit.

    Not optional. A bare `memref.alloc` is uninitialized, and an unfilled input
    makes even the *clean* kernel report `value-changing` -- which silently invalidates
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


def _device_in(
    e: Emitter, dims: Sequence[int | None], resolved: Sequence[int], mtype: str,
    host: str, tok: str, name: str,
) -> tuple[str, str]:
    """Allocate a device buffer and copy `host` into it; returns `(buffer, tok)`.

    Used for every input *and* for the output buffer: the kernel may read the
    output before writing it (softmax normalizes in place), so the device copy has
    to be seeded from the same zeroed host buffer the CPU surface reads.
    """
    d, t = _gpu_alloc(e, dims, resolved, mtype, tok, name)
    t = _gpu_memcpy(e, t, d, host, mtype)
    return d, t


# --------------------------------------------------------------------------
# M4 -- loop-bound invalidity (class `LoopBound`, families M4-a..M4-g)
# --------------------------------------------------------------------------
#
# M4 is the control for M1: it is *not* a shape class. Every family makes the
# iteration space of an otherwise-clean kernel empty (or non-terminating), so the
# body simply never executes and the output is silently unwritten. The memref
# surface authors its own `scf.for` bounds, so each family is realised by editing
# the bound at the one place the loop is emitted (`_m4_triple`); M4-g instead
# derives the extent from a 0-length view so the clean bound rule already yields
# 0. This mirrors triton, which realises M4 in its generator rather than through
# `mutate.apply` -- M4 is not an index perturbation.

_M4_KINDS: dict[str, str] = {
    "M4.1": "m4-zero-bound",
    "M4.6": "m4-negative-bound",
    "M4.3": "m4-runtime-zero-bound",
    "M1.7": "m4-reversed-bound",
    "M4.5": "m4-zero-step",
    "M4.8": "m4-degenerate-pad",
}


def m4_structural(case: C.Case, mut: C.Mutation) -> Structural:
    """Structural directive for an M4 loop-bound spec on the memref surface.

    `spec_id` (not `spec`) keys the table because M1.7 is re-homed to family M4-e
    by the taxonomy and has to route here, while its M1 siblings stay in
    `mutate._m1_structural`. The axis is always the trailing one: every M4 defect
    is on the boundary loop of the last axis.
    """
    kind = _M4_KINDS.get(mut.spec_id)
    if kind is None:
        raise KeyError(f"no M4 realisation for spec {mut.spec_id!r}")
    name = "inp" if "inp" in case.dims else "lhs"
    return Structural(kind=kind, axis=len(case.dims[name]) - 1)


def _m4_runtime_zero(e: Emitter) -> str:
    """An `index` that is 0 only at runtime (family M4-c / spec M4.3).

    Loaded from a device buffer the prologue seeded with 0. A live SSA value --
    rather than the literal 0 of M4-a -- is what keeps the empty iteration space
    a *runtime* property: neither the verifier nor `canonicalize` can prove the
    loop empty, so the mutant lowers and runs, silently writing nothing.
    `memref.dim` of a static memref folds to a constant, and `x - x` folds to 0,
    so neither can carry this family; a device load has no such fold.
    """
    buf = getattr(e, "_m4_zero_buf", None)
    if buf is None:
        raise NotExpressible(
            "runtime-zero bound needs the M4 prologue: call "
            "`_m4_zero_prologue(e, st, tok)` before the launch")
    c0 = _cidx(e, 0)
    v = e.new("z")
    e.emit(f"{v} = memref.load {buf}[{c0}] : memref<1xi64>")
    r = e.new("z")
    e.emit(f"{r} = arith.index_cast {v} : i64 to index")
    return r


def _m4_zero_prologue(e: Emitter, st: Structural | None, tok: str) -> str:
    """Seed the device buffer M4-c loads its bound from; returns the new token.

    A no-op for every other directive. The buffer is one `i64` in device memory
    (not a host/managed allocation, which compute-sanitizer cannot see), copied
    in from a zeroed host constant, so the value the kernel loads is 0.
    """
    if st is None or st.kind != "m4-runtime-zero-bound":
        return tok
    if getattr(e, "_m4_zero_buf", None) is not None:
        return tok
    dims, resolved = (1,), (1,)
    # `_alloc` derives an f32 memref; M4-c's carrier is an `i64`, so the host
    # buffer is spelled directly. Device allocation/memcpy still go through the
    # shared helpers, which take the element type explicitly.
    mtype = C.memref_type(dims, dtype="i64")
    host = e.new("zbuf")
    e.emit(f"{host} = memref.alloc() : {mtype}")
    z = e.new("c")
    e.emit(f"{z} = arith.constant 0 : i64")
    c0 = e.new("c")
    e.emit(f"{c0} = arith.constant 0 : index")
    e.emit(f"memref.store {z}, {host}[{c0}] : {mtype}")
    dev, tok = _device_in(e, dims, resolved, mtype, host, tok, "dzero")
    e._m4_zero_buf = dev
    return tok


def _m4_degenerate_view(
    e: Emitter, buf: str, buf_t: str, dims: Sequence[int | None], axis: int
) -> tuple[str, str]:
    """A rank-preserving subview of `buf` whose `axis` extent is 0 (family M4-g).

    Every other size is the parent's true extent, so the descriptor is legal; the
    defect is that the caller reads its loop extent from this view, whose `axis`
    size is 0, and the loop body therefore never runs. Unlike M1's
    symbolic-offset subview this one is not an out-of-bounds descriptor -- it is
    exactly 0 long -- so the mutant stays silent instead of tripping RTV.

    The degenerate axis also takes `stride 0`. RTV's subview check asserts
    `0 <= offset + (size - 1) * stride < dim`; with `size = 0` and a nonzero
    stride the `(size - 1) * stride` term is `-stride` and the check fails on a
    *legal* empty slice (`RuntimeOpVerification.cpp:290`). Neutralising the
    stride keeps the last-position term equal to the in-bounds offset, so the
    empty view verifies. The stride is immaterial to an empty range -- the extent
    read from the view is what carries the mutation -- so this does not change
    the defect.
    """
    nd = len(dims)
    c0 = e.new("c")
    e.emit(f"{c0} = arith.constant 0 : index")
    c1 = e.new("c")
    e.emit(f"{c1} = arith.constant 1 : index")
    offsets = [c0] * nd
    sizes = [_dim(e, buf, buf_t, k) for k in range(nd)]
    sizes[axis] = c0
    strides = [c1] * nd
    strides[axis] = c0
    vt = _view_type(nd)
    sv = e.new("sv")
    e.emit(
        f"{sv} = memref.subview {buf}[{', '.join(offsets)}] "
        f"[{', '.join(sizes)}] [{', '.join(strides)}] : {buf_t} to {vt}"
    )
    return sv, vt


def _m4_triple(
    e: Emitter, st: Structural | None, lo: str, hi: str, step: str, ext: str
) -> tuple[str, str, str]:
    """Override a loop's `(lower, upper, step)` to realise an M4 defect.

    The clean triple passes through untouched for every non-M4 directive, so the
    M1 perturbations are unaffected. `m4-degenerate-pad` is deliberately absent:
    its empty iteration space comes from `ext` being 0 (the extent is read from a
    0-length view), and `_round_up(0, tile)` is already 0, so the clean triple is
    itself empty.

    `m4-reversed-bound` reads `lo = ext, hi = 0` -- an upper bound strictly below
    the lower -- rather than keeping `lo` and zeroing `hi`. Both are empty for the
    battery's extents, but `lo = ext, hi = 0` stays empty even when the block's
    starting thread `tx` exceeds the bound, which the reversed spelling states
    directly.
    """
    if st is None or not st.kind.startswith("m4-"):
        return lo, hi, step
    if st.kind == "m4-zero-bound":
        return lo, _cidx(e, 0), step
    if st.kind == "m4-negative-bound":
        return lo, _cidx(e, -1), step
    if st.kind == "m4-runtime-zero-bound":
        return lo, _m4_runtime_zero(e), step
    if st.kind == "m4-reversed-bound":
        return ext, _cidx(e, 0), step
    if st.kind == "m4-zero-step":
        return lo, hi, _cidx(e, 0)
    return lo, hi, step


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

    # Extents and the rounded-up bound are read off the *host* buffer and captured
    # by the launch as index operands, so the kernel body never needs `affine` and
    # the guard compares against a live extent rather than a literal.
    ext_src, ext_t = inp, inp_t
    if st is not None and st.kind == "m4-degenerate-pad":
        ext_src, ext_t = _m4_degenerate_view(e, inp, inp_t, inp_dims, axis)
    ext = _dim(e, ext_src, ext_t, axis)
    bound = _round_up(e, ext, _tile_for(inp_r[axis]))
    obounds = [_dim(e, inp, inp_t, k) for k in range(axis)]

    tok = e.new("tok")
    e.emit(f"{tok} = gpu.wait async")
    d_inp, tok = _device_in(e, inp_dims, inp_r, inp_t, inp, tok, "dinp")
    d_out, tok = _device_in(e, out_dims, out_r, out_t, out, tok, "dout")
    tok = _m4_zero_prologue(e, st, tok)

    nblock = _cidx(e, _GPU_BLOCK)
    grid = _prod(e, obounds)
    z = e.new("c")
    e.emit(f"{z} = arith.constant 0.0 : f32")
    with _launch(e, grid, nblock) as (bx, tx):
        outer = _decompose(e, bx, obounds)
        rd, rd_t = d_inp, inp_t
        if st is not None and st.kind == "subview-symbolic":
            rd, rd_t = _subview(e, d_inp, inp_t, inp_dims, outer[0])
        with _guarded_axis_loop(e, tx, bound, nblock, ext, st) as j:
            ridx = _perturb(e, outer + [j], st, axis)
            v = _read(e, rd, rd_t, ridx, st)
            # `arith.maxf` was renamed in LLVM 21; `maximumf` propagates NaN like
            # numpy.maximum, which is what the reference oracle uses.
            r = e.new("r")
            e.emit(f"{r} = arith.maximumf {v}, {z} : f32")
            widx = _write_perturb(e, outer + [j], st, nd - 1)
            e.emit(f"memref.store {r}, {d_out}[{', '.join(widx)}] : {out_t}")

    tok = _gpu_memcpy(e, tok, out, d_out, out_t)
    _gpu_wait(e, tok)
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

    ext_src, ext_t = inp, inp_t
    if st is not None and st.kind == "m4-degenerate-pad":
        ext_src, ext_t = _m4_degenerate_view(e, inp, inp_t, inp_dims, axis)
    ext = _dim(e, ext_src, ext_t, axis)
    bound = _round_up(e, ext, _tile_for(inp_r[axis]))
    obounds = [_dim(e, inp, inp_t, k) for k in range(axis)]

    tok = e.new("tok")
    e.emit(f"{tok} = gpu.wait async")
    d_inp, tok = _device_in(e, inp_dims, inp_r, inp_t, inp, tok, "dinp")
    d_out, tok = _device_in(e, out_dims, out_r, out_t, out, tok, "dout")
    tok = _m4_zero_prologue(e, st, tok)

    nblock = _cidx(e, _GPU_BLOCK)
    grid = _prod(e, obounds)
    with _launch(e, grid, nblock) as (bx, tx):
        outer = _decompose(e, bx, obounds)
        rd, rd_t = d_inp, inp_t
        if st is not None and st.kind == "subview-symbolic":
            rd, rd_t = _subview(e, d_inp, inp_t, inp_dims, outer[0])
        with _guarded_axis_loop(e, tx, bound, nblock, ext, st) as j:
            ridx = _perturb(e, outer + [j], st, axis)
            v = _read(e, rd, rd_t, ridx, st)
            # The store reverses the loop indices, so the tiled axis (j) is
            # coordinate 0 here; overlap-write wraps that coordinate.
            widx = _write_perturb(e, list(reversed(outer + [j])), st, 0)
            e.emit(f"memref.store {v}, {d_out}[{', '.join(widx)}] : {out_t}")

    tok = _gpu_memcpy(e, tok, out, d_out, out_t)
    _gpu_wait(e, tok)
    return out, out_t, out_r


def _low_softmax(e: Emitter, case: C.Case, st: Structural | None):
    """Row-wise softmax over the trailing axis, rank-generic.

    Three passes per row (max, exp+sum, normalize), the standard formulation and
    the one `compose.reference` implements. Each pass is block-parallel over the
    row's trailing axis and folded with a `gpu.all_reduce`, which stages the
    per-thread partials through workgroup shared memory.
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

    ext_src, ext_t = inp, inp_t
    if st is not None and st.kind == "m4-degenerate-pad":
        ext_src, ext_t = _m4_degenerate_view(e, inp, inp_t, inp_dims, axis)
    ext = _dim(e, ext_src, ext_t, axis)
    bound = _round_up(e, ext, _tile_for(inp_r[axis]))
    obounds = [_dim(e, inp, inp_t, k) for k in range(axis)]

    ninf = e.new("c")
    e.emit(f"{ninf} = arith.constant {_NEG_INF} : f32")
    zero = e.new("c")
    e.emit(f"{zero} = arith.constant 0.0 : f32")

    tok = e.new("tok")
    e.emit(f"{tok} = gpu.wait async")
    d_inp, tok = _device_in(e, inp_dims, inp_r, inp_t, inp, tok, "dinp")
    d_out, tok = _device_in(e, out_dims, out_r, out_t, out, tok, "dout")
    tok = _m4_zero_prologue(e, st, tok)

    nblock = _cidx(e, _GPU_BLOCK)
    grid = _prod(e, obounds)
    with _launch(e, grid, nblock) as (bx, tx):
        outer = _decompose(e, bx, obounds)
        rd, rd_t = d_inp, inp_t
        if st is not None and st.kind == "subview-symbolic":
            rd, rd_t = _subview(e, d_inp, inp_t, inp_dims, outer[0])

        # --- pass 1: row max -------------------------------------------------
        def _maxbody(e: Emitter, ridx: list[str], acc: str, iv: str) -> str:
            v = _read(e, rd, rd_t, ridx, st)
            m = e.new("t")
            e.emit(f"{m} = arith.maximumf {acc}, {v} : f32")
            return m

        mx = _block_reduce(
            e, outer, ext, bound, axis, ninf, st, tx, nblock, _maxbody,
            "arith.maximumf",
        )

        # --- pass 2: exp and row sum -----------------------------------------
        def _expbody(e: Emitter, ridx: list[str], acc: str, iv: str) -> str:
            v = _read(e, rd, rd_t, ridx, st)
            d = e.new("t")
            e.emit(f"{d} = arith.subf {v}, {mx} : f32")
            ex = e.new("t")
            e.emit(f"{ex} = math.exp {d} : f32")
            # Written at the loop's own index, not the perturbed read index, so
            # the defect stays isolated in the read. The store is still inside the
            # guard, so dropping the guard (M1.1) overruns the destination too.
            widx = list(outer) + [iv]
            e.emit(f"memref.store {ex}, {d_out}[{', '.join(widx)}] : {out_t}")
            ns = e.new("t")
            e.emit(f"{ns} = arith.addf {acc}, {ex} : f32")
            return ns

        sm = _block_reduce(
            e, outer, ext, bound, axis, zero, st, tx, nblock, _expbody,
            "arith.addf",
        )

        # Pass 2 wrote the exponentials; pass 3 re-reads them, so the block must
        # rendezvous before normalizing.
        e.emit("gpu.barrier")

        # --- pass 3: normalize (unguarded; walks the true extent) ------------
        k = e.new("k")
        lo3, hi3, st3 = _m4_triple(e, st, tx, ext, nblock, ext)
        e.emit(f"scf.for {k} = {lo3} to {hi3} step {st3} {{")
        e.indent += 1
        widx3 = _write_perturb(e, list(outer) + [k], st, axis)
        ev = e.new("v")
        e.emit(f"{ev} = memref.load {d_out}[{', '.join(widx3)}] : {out_t}")
        nv = e.new("t")
        e.emit(f"{nv} = arith.divf {ev}, {sm} : f32")
        e.emit(f"memref.store {nv}, {d_out}[{', '.join(widx3)}] : {out_t}")
        e.indent -= 1
        e.emit("}")

    tok = _gpu_memcpy(e, tok, out, d_out, out_t)
    _gpu_wait(e, tok)
    return out, out_t, out_r


def _low_layer_norm(e: Emitter, case: C.Case, st: Structural | None):
    """Layer normalization over the trailing axis, rank-generic.

    The two reductions (row sum, then sum of squared deviations) are each folded
    with a `gpu.all_reduce`, so the workgroup shared-memory path carries the
    statistics just as it does for softmax.
    """
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

    ext_src, ext_t = lhs, lhs_t
    if st is not None and st.kind == "m4-degenerate-pad":
        ext_src, ext_t = _m4_degenerate_view(e, lhs, lhs_t, lhs_dims, axis)
    ext = _dim(e, ext_src, ext_t, axis)
    bound = _round_up(e, ext, _tile_for(lhs_r[axis]))
    obounds = [_dim(e, lhs, lhs_t, k) for k in range(axis)]

    zero = e.new("c")
    e.emit(f"{zero} = arith.constant 0.0 : f32")
    # The row length as f32, computed once on the host and captured.
    nf = e.new("n")
    e.emit(f"{nf} = arith.index_cast {ext} : index to i64")
    nf32 = e.new("nf")
    e.emit(f"{nf32} = arith.sitofp {nf} : i64 to f32")

    tok = e.new("tok")
    e.emit(f"{tok} = gpu.wait async")
    d_lhs, tok = _device_in(e, lhs_dims, lhs_r, lhs_t, lhs, tok, "dlhs")
    d_scale, tok = _device_in(e, sc_dims, sc_r, scale_t, scale, tok, "dscale")
    d_bias, tok = _device_in(e, bi_dims, bi_r, bias_t, bias, tok, "dbias")
    d_out, tok = _device_in(e, out_dims, out_r, out_t, out, tok, "dout")
    tok = _m4_zero_prologue(e, st, tok)

    nblock = _cidx(e, _GPU_BLOCK)
    grid = _prod(e, obounds)
    with _launch(e, grid, nblock) as (bx, tx):
        outer = _decompose(e, bx, obounds)
        rd, rd_t = d_lhs, lhs_t
        if st is not None and st.kind == "subview-symbolic":
            rd, rd_t = _subview(e, d_lhs, lhs_t, lhs_dims, outer[0])

        # --- mean ------------------------------------------------------------
        def _sumbody(e: Emitter, ridx: list[str], acc: str, iv: str) -> str:
            v = _read(e, rd, rd_t, ridx, st)
            a = e.new("t")
            e.emit(f"{a} = arith.addf {acc}, {v} : f32")
            return a

        tot = _block_reduce(
            e, outer, ext, bound, axis, zero, st, tx, nblock, _sumbody,
            "arith.addf",
        )

        mean = e.new("mean")
        e.emit(f"{mean} = arith.divf {tot}, {nf32} : f32")

        # --- variance --------------------------------------------------------
        def _sqbody(e: Emitter, ridx: list[str], acc: str, iv: str) -> str:
            v = _read(e, rd, rd_t, ridx, st)
            d = e.new("t")
            e.emit(f"{d} = arith.subf {v}, {mean} : f32")
            m = e.new("t")
            e.emit(f"{m} = arith.mulf {d}, {d} : f32")
            a = e.new("t")
            e.emit(f"{a} = arith.addf {acc}, {m} : f32")
            return a

        sq = _block_reduce(
            e, outer, ext, bound, axis, zero, st, tx, nblock, _sqbody,
            "arith.addf",
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
        lo3, hi3, st3 = _m4_triple(e, st, tx, ext, nblock, ext)
        e.emit(f"scf.for {k} = {lo3} to {hi3} step {st3} {{")
        e.indent += 1
        ridx_k = _perturb(e, list(outer) + [k], st, axis)
        xv = _read(e, rd, rd_t, ridx_k, st)
        d2 = e.new("t")
        e.emit(f"{d2} = arith.subf {xv}, {mean} : f32")
        n2 = e.new("t")
        e.emit(f"{n2} = arith.divf {d2}, {sd} : f32")
        sc = e.new("t")
        e.emit(f"{sc} = memref.load {d_scale}[{k}] : {scale_t}")
        m1 = e.new("t")
        e.emit(f"{m1} = arith.mulf {n2}, {sc} : f32")
        bi = e.new("t")
        e.emit(f"{bi} = memref.load {d_bias}[{k}] : {bias_t}")
        m2 = e.new("t")
        e.emit(f"{m2} = arith.addf {m1}, {bi} : f32")
        widx = _write_perturb(e, list(outer) + [k], st, axis)
        e.emit(f"memref.store {m2}, {d_out}[{', '.join(widx)}] : {out_t}")
        e.indent -= 1
        e.emit("}")

    tok = _gpu_memcpy(e, tok, out, d_out, out_t)
    _gpu_wait(e, tok)
    return out, out_t, out_r


LOW_EMITTERS = {
    "relu": _low_relu,
    "transpose": _low_transpose,
    "softmax": _low_softmax,
    "layer_normalization": _low_layer_norm,
    # M1-f carrier hosts: the same operator bodies, distinguishable only by the
    # narrow-carrier structural directive the battery attaches to them.
    "relu_carrier": _low_relu,
    "transpose_carrier": _low_transpose,
    "softmax_carrier": _low_softmax,
    "layer_normalization_carrier": _low_layer_norm,
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
