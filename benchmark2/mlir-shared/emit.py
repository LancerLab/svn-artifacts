#!/usr/bin/env python3
"""MLIR emitter for the benchmark2 MLIR lanes.

Turns a `compose.Case` into a self-contained MLIR module whose `@main` returns
i32: `0` when the kernel's output matches the numpy reference, `1` otherwise.

Why a checksum oracle rather than printing outputs:

* it works at *any* input size, so the same kernel serves `--small` and `--full`
  without change;
* it answers the ref-check gate and the §7 manifest oracle (`corrupts` vs `noop`)
  with a single value, so a `noop` false success cannot slip through;
* two independent checksums (sum and sum-of-squares) are compared, which makes an
  accidental collision far less likely than a single sum.

Inputs are generated *inside* the kernel from the deterministic formula in
`compose.input_values`, using `linalg.index`. That keeps the emitted module small
even for full-size inputs -- no thousands-of-constants dump.

The emitter is deliberately **mutation-agnostic**: `mutate.apply` perturbs the
`Case` dims and the resulting type inconsistency is what produces the defect.
Only structural defects (`mutate.Structural`) are passed in explicitly, because
they change the write pattern rather than a type.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

import numpy as np

import compose as C
from mutate import Structural

# Checksum-comparison tolerances live in `compose.tolerance`, which is *relative*
# to each checksum's magnitude. An absolute per-element constant used to sit here;
# it was removed because it was ~1000x too loose and let a real defect through.


def f32_lit(v: float) -> str:
    """Render a Python float as an MLIR f32 literal that the lexer accepts.

    MLIR's lexer requires a decimal point in a float literal. Python's `repr`
    omits it whenever the shortest round-trip form uses an exponent, so
    `repr(1e-5)` is `"1e-05"` and `repr(2e20)` is `"2e+20"` -- both of which
    mlir-opt rejects with `custom op 'e' is unknown`, because it never sees a
    float and falls back to parsing `e` as an operation name.

    This is not a corner case: the oracle bakes in `ref_sum`, `ref_sq` and the
    tolerance, and a large reduction can easily produce a checksum whose shortest
    repr is exponent form. Without this helper the failure would surface as a
    parse error in a lane that otherwise looks healthy.
    """
    s = repr(float(v))
    if "." not in s and "inf" not in s and "nan" not in s:
        s = s.replace("e", ".0e").replace("E", ".0E")
    return s


class Emitter:
    """Emits MLIR text with a simple SSA-name counter.

    Every op is tagged with a source `loc`. This is not cosmetic: RTV bakes the
    location into each generated assert message (`Location: loc("kernel")`), which
    is the only way to attribute an instrumented guard to the *kernel under audit*
    rather than to this harness's own checksum oracle. Counting oracle guards as
    kernel guards would inflate S9 (remainder = Σ unconditional_guards) with
    harness artifacts.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.n = 0
        self.indent = 1
        self.loc = "kernel"

    def new(self, hint: str = "v") -> str:
        self.n += 1
        return f"%{hint}{self.n}"

    # Lines that are complete op statements and therefore take a trailing `loc`.
    # Anything else -- an attribute line inside an op's `{...}` such as
    # `iterator_types = [...]`, a block-argument header `^bb0(...)`, an op header
    # that opens a region, or a bare `}` -- cannot carry one.
    #
    # The memref-level prefixes (`affine.store`, `affine.yield`, `memref.store`,
    # `memref.copy`) are here because those ops produce no SSA value, so their
    # statements do not start with `%` and would otherwise go untagged. An
    # untagged assert silently loses its kernel/oracle attribution, which is what
    # keeps S9 honest. (`affine.apply` is absent because it *does* yield a value
    # and is caught by the `%` check.)
    _BARE_OPS = (
        "linalg.yield", "linalg.fill", "return", "func.", "cf.", "scf.",
        "affine.store", "affine.yield",
        "memref.store", "memref.copy", "memref.dealloc",
    )

    def emit(self, s: str = "") -> None:
        """Emit one line, tagging it with the current location when it is an op."""
        if not s:
            self.lines.append("")
            return
        body = "  " * self.indent + s
        t = s.strip()
        # A lone `}` closes a func or module; it takes no location. A `} -> T`
        # closes a region-bearing op and does, since that is where MLIR puts it.
        is_op = (
            (t.startswith("%") or t.startswith(self._BARE_OPS)
             or (t.startswith("}") and t != "}"))
        )
        opens_region = t.endswith("{")
        if is_op and not opens_region:
            self.lines.append(f'{body} loc("{self.loc}")')
        else:
            self.lines.append(body)

    def raw(self, s: str = "") -> None:
        """Emit one line verbatim, with no location tag."""
        self.lines.append("  " * self.indent + s if s else "")

    def end(self, s: str) -> None:
        """Emit an op-terminating line, tagged with the current location."""
        self.emit(f"{s}")

    @contextmanager
    def location(self, name: str) -> Iterator[None]:
        """Set the source location for ops emitted inside the block."""
        prev = self.loc
        self.loc = name
        try:
            yield
        finally:
            self.loc = prev

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


# --------------------------------------------------------------------------
# iterator_types helpers
# --------------------------------------------------------------------------
# Python 3.10 forbids backslashes inside f-string expressions, so the quoted
# iterator names are built here rather than inline.

_PAR = '"parallel"'
_RED = '"reduction"'


def iters_all_parallel(nd: int) -> str:
    return ",".join([_PAR] * nd)


def iters_trailing_reduction(nd: int) -> str:
    """`nd-1` parallel dims then one reduction (softmax / layer_norm row ops)."""
    return ",".join([_PAR] * (nd - 1) + [_RED])


def iters_all_reduction(nd: int) -> str:
    return ",".join([_RED] * nd)


# --------------------------------------------------------------------------
# Shape helpers
# --------------------------------------------------------------------------


def concrete(
    dims: tuple[int | None, ...], dyn: dict[str, int], name: str
) -> tuple[int, ...]:
    """Resolve a declared shape, binding dynamic extents per manifest §5.2."""
    return tuple(d if d is not None else dyn[f"{name}.{k}"] for k, d in enumerate(dims))


def emit_empty(
    e: Emitter,
    dims: tuple[int | None, ...],
    resolved: Sequence[int],
    dtype: str = "f32",
) -> str:
    """Emit `tensor.empty`, supplying operands for every dynamic extent.

    Per manifest §5.2 rule 3 the dynamic extents are bound to concrete small
    constants at the entry, so each `?` becomes an `arith.constant`. Omitting
    these operands is a verifier error ("incorrect number of dynamic sizes").

    `dims` is the declared shape (may contain `None`); `resolved` is the same rank
    with every extent bound. Taking the resolved shape rather than looking it up
    by operand name lets this work for derived tensors too (a softmax row tensor,
    a transposed output) whose dynamic axes belong to no declared operand.
    """
    if len(dims) != len(resolved):
        raise ValueError(f"dims {dims} and resolved {resolved} differ in rank")
    ttype = C.tensor_type(dims, dtype)
    args = []
    for k, d in enumerate(dims):
        if d is None:
            v = e.new("d")
            e.emit(f"{v} = arith.constant {resolved[k]} : index")
            args.append(v)
    ssa = e.new("empty")
    if args:
        e.emit(f"{ssa} = tensor.empty({', '.join(args)}) : {ttype}")
    else:
        e.emit(f"{ssa} = tensor.empty() : {ttype}")
    return ssa


# --------------------------------------------------------------------------
# Formula fill: flat element i -> ((i*37) % 13) - 6, plus a per-operand offset
# --------------------------------------------------------------------------


def _flat_index(
    e: Emitter, idx: Sequence[str], extents: Sequence[str], offset: int = 0
) -> str:
    """Row-major flat index (an `index` SSA value) from per-dim index values.

    `extents` are the *true* extents of the buffer being walked, so the flat index
    agrees with numpy's C-order ravel regardless of whether the shape is static or
    dynamic. Shared by the input formula and the oracle's position weights, which
    must both derive position the same way or the two oracles disagree.
    """
    nd = len(idx)
    strides: list[str] = [""] * nd
    acc = e.new("c")
    e.emit(f"{acc} = arith.constant 1 : index")
    for k in range(nd - 1, -1, -1):
        strides[k] = acc
        if k > 0:
            nxt = e.new("s")
            e.emit(f"{nxt} = arith.muli {acc}, {extents[k]} : index")
            acc = nxt

    flat = e.new("c")
    e.emit(f"{flat} = arith.constant {offset} : index")
    for k in range(nd):
        t = e.new("t")
        e.emit(f"{t} = arith.muli {idx[k]}, {strides[k]} : index")
        flat2 = e.new("t")
        e.emit(f"{flat2} = arith.addi {flat}, {t} : index")
        flat = flat2
    return flat


def _position_weight(e: Emitter, flat: str) -> str:
    """Flat position -> weight `((i * _W_MUL) % _W_MOD) - _W_SUB` as f32.

    Mirrors `compose.position_weights` exactly (i64 arithmetic, then widened), so
    the weight baked into the kernel and the one the numpy oracle computes are the
    same number. This is what makes the third checksum order-sensitive: see
    `compose.checksums`.
    """
    i64 = e.new("i")
    e.emit(f"{i64} = arith.index_cast {flat} : index to i64")
    cm = e.new("c")
    e.emit(f"{cm} = arith.constant {C._W_MUL} : i64")
    cd = e.new("c")
    e.emit(f"{cd} = arith.constant {C._W_MOD} : i64")
    cs = e.new("c")
    e.emit(f"{cs} = arith.constant {C._W_SUB} : i64")
    m = e.new("t")
    e.emit(f"{m} = arith.muli {i64}, {cm} : i64")
    r = e.new("t")
    e.emit(f"{r} = arith.remsi {m}, {cd} : i64")
    s = e.new("t")
    e.emit(f"{s} = arith.subi {r}, {cs} : i64")
    w = e.new("t")
    e.emit(f"{w} = arith.sitofp {s} : i64 to f32")
    return w


def _flat_index_body(
    e: Emitter, idx: Sequence[str], extents: Sequence[str], offset: int
) -> str:
    """Emit arithmetic computing the input value at a position from `linalg.index`."""
    flat = _flat_index(e, idx, extents, offset)

    i64 = e.new("i")
    e.emit(f"{i64} = arith.index_cast {flat} : index to i64")
    c37 = e.new("c")
    e.emit(f"{c37} = arith.constant {C._MUL} : i64")
    c13 = e.new("c")
    e.emit(f"{c13} = arith.constant {C._MOD} : i64")
    c6 = e.new("c")
    e.emit(f"{c6} = arith.constant {C._SUB} : i64")
    m = e.new("t")
    e.emit(f"{m} = arith.muli {i64}, {c37} : i64")
    r = e.new("t")
    e.emit(f"{r} = arith.remsi {m}, {c13} : i64")
    s = e.new("t")
    e.emit(f"{s} = arith.subi {r}, {c6} : i64")
    v = e.new("t")
    e.emit(f"{v} = arith.sitofp {s} : i64 to f32")
    return v


def emit_formula_tensor(
    e: Emitter,
    dims: tuple[int | None, ...],
    resolved: Sequence[int],
    offset: int = 0,
    name: str = "t",
) -> str:
    """Emit a `linalg.generic` filling a tensor with the deterministic formula.

    `dims` is the *declared* shape and `resolved` the bound extents used as the
    stride basis. Callers pass `resolved` derived from the same declared shape, so
    a mutation that declares a wrong extent produces a tensor that is genuinely
    that size and genuinely filled at that size -- the defect then lives in the
    *op* that consumes it with a mismatched partner, which is what the M2 specs
    describe, rather than in an unrelated perturbation of the value stream.
    """
    ttype = C.tensor_type(dims)
    nd = len(dims)
    empty = emit_empty(e, dims, resolved)

    dim_names = ",".join(f"d{i}" for i in range(nd))
    idmap = f"affine_map<({dim_names}) -> ({dim_names})>" if nd else "affine_map<() -> ()>"

    res = e.new(name)
    e.emit(f"{res} = linalg.generic {{")
    e.indent += 1
    e.emit(f"indexing_maps = [{idmap}],")
    e.emit(f"iterator_types = [{iters_all_parallel(nd)}]")
    e.indent -= 1
    e.emit(f"}} outs({empty} : {ttype}) {{")
    e.indent += 1
    e.emit("^bb0(%o: f32):")
    idx = []
    for k in range(nd):
        v = e.new("i")
        e.emit(f"{v} = linalg.index {k} : index")
        idx.append(v)
    ext = []
    for k in range(nd):
        v = e.new("d")
        e.emit(f"{v} = arith.constant {resolved[k]} : index")
        ext.append(v)
    val = _flat_index_body(e, idx, ext, offset)
    e.emit(f"linalg.yield {val} : f32")
    e.indent -= 1
    e.emit(f"}} -> {ttype}")
    return res


# --------------------------------------------------------------------------
# Reductions (checksums) and the oracle tail
# --------------------------------------------------------------------------


def emit_reduce(e: Emitter, src: str, src_type: str, nd: int, square: bool) -> str:
    """Full reduction of `src` to a scalar `tensor<f32>`: sum, or sum of squares."""
    init = e.new("init")
    e.emit(f"{init} = tensor.empty() : tensor<f32>")
    z = e.new("c")
    e.emit(f"{z} = arith.constant 0.0 : f32")
    filled = e.new("f")
    e.emit(
        f"{filled} = linalg.fill ins({z} : f32) outs({init} : tensor<f32>)"
        f" -> tensor<f32>"
    )

    dims = ",".join(f"d{i}" for i in range(nd))
    inmap = f"affine_map<({dims}) -> ({dims})>" if nd else "affine_map<() -> ()>"
    outmap = f"affine_map<({dims}) -> ()>" if nd else "affine_map<() -> ()>"

    res = e.new("r")
    e.emit(f"{res} = linalg.generic {{")
    e.indent += 1
    e.emit(f"indexing_maps = [{inmap}, {outmap}],")
    e.emit(f"iterator_types = [{iters_all_reduction(nd)}]")
    e.indent -= 1
    e.emit(f"}} ins({src} : {src_type}) outs({filled} : tensor<f32>) {{")
    e.indent += 1
    e.emit("^bb0(%x: f32, %acc: f32):")
    if square:
        sq = e.new("t")
        e.emit(f"{sq} = arith.mulf %x, %x : f32")
        add = e.new("t")
        e.emit(f"{add} = arith.addf %acc, {sq} : f32")
        e.emit(f"linalg.yield {add} : f32")
    else:
        add = e.new("t")
        e.emit(f"{add} = arith.addf %acc, %x : f32")
        e.emit(f"linalg.yield {add} : f32")
    e.indent -= 1
    e.emit("} -> tensor<f32>")
    return res


def emit_extract0d(e: Emitter, t: str) -> str:
    v = e.new("s")
    e.emit(f"{v} = tensor.extract {t}[] : tensor<f32>")
    return v


def emit_reduce_weighted(e: Emitter, src: str, src_type: str, nd: int) -> str:
    """Full reduction of `src` to `sum(v[i] * w(flat_i))` as a scalar `tensor<f32>`.

    This is the order-sensitive companion to `emit_reduce`. It needs the position
    of each element, which a plain reduction hides, so the body reads it back with
    `linalg.index` and derives the row-major flat index from the buffer's *true*
    extents (`tensor.dim`), not from literals -- that way a mutant whose declared
    shape disagrees with the buffer it actually produced cannot have its position
    arithmetic silently agree with the clean kernel's.

    `linalg.index` is legal in a reduction body and survives the pinned linalg
    pipeline (verified separately); the extents are captured from outside the
    region, which `linalg.generic` permits.
    """
    init = e.new("init")
    e.emit(f"{init} = tensor.empty() : tensor<f32>")
    z = e.new("c")
    e.emit(f"{z} = arith.constant 0.0 : f32")
    filled = e.new("f")
    e.emit(
        f"{filled} = linalg.fill ins({z} : f32) outs({init} : tensor<f32>)"
        f" -> tensor<f32>"
    )

    ext: list[str] = []
    if nd:
        for k in range(nd):
            ck = e.new("c")
            e.emit(f"{ck} = arith.constant {k} : index")
            d = e.new("d")
            e.emit(f"{d} = tensor.dim {src}, {ck} : {src_type}")
            ext.append(d)

    dims = ",".join(f"d{i}" for i in range(nd))
    inmap = f"affine_map<({dims}) -> ({dims})>" if nd else "affine_map<() -> ()>"
    outmap = f"affine_map<({dims}) -> ()>" if nd else "affine_map<() -> ()>"

    res = e.new("r")
    e.emit(f"{res} = linalg.generic {{")
    e.indent += 1
    e.emit(f"indexing_maps = [{inmap}, {outmap}],")
    e.emit(f"iterator_types = [{iters_all_reduction(nd)}]")
    e.indent -= 1
    e.emit(f"}} ins({src} : {src_type}) outs({filled} : tensor<f32>) {{")
    e.indent += 1
    e.emit("^bb0(%x: f32, %acc: f32):")
    idx = []
    for k in range(nd):
        v = e.new("i")
        e.emit(f"{v} = linalg.index {k} : index")
        idx.append(v)
    flat = _flat_index(e, idx, ext)
    w = _position_weight(e, flat)
    p = e.new("t")
    e.emit(f"{p} = arith.mulf %x, {w} : f32")
    add = e.new("t")
    e.emit(f"{add} = arith.addf %acc, {p} : f32")
    e.emit(f"linalg.yield {add} : f32")
    e.indent -= 1
    e.emit("} -> tensor<f32>")
    return res


def emit_oracle(e: Emitter, out_ssa: str, out_type: str, nd: int, ref: np.ndarray) -> str:
    """Compare three checksums against the baked-in reference; return the i32 flag.

    The third (position-weighted) check exists because the first two are
    permutation-invariant and so cannot see an M1 defect that reorders values; see
    `compose.checksums`. Each check gets its own tolerance via `compose.tolerance`,
    which is *relative* to that checksum's magnitude -- an absolute budget let a
    real softmax drop-mask defect slip through as `noop`.
    """
    ref_sum, ref_sq, ref_w = C.checksums(ref)
    nel = int(ref.size)
    tol_s = C.tolerance(nel, ref_sum)
    tol_q = C.tolerance(nel, ref_sq)
    tol_w = C.tolerance(nel, ref_w)

    s_t = emit_reduce(e, out_ssa, out_type, nd, square=False)
    q_t = emit_reduce(e, out_ssa, out_type, nd, square=True)
    w_t = emit_reduce_weighted(e, out_ssa, out_type, nd)
    s = emit_extract0d(e, s_t)
    q = emit_extract0d(e, q_t)
    wv = emit_extract0d(e, w_t)

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
    res = e.new("r")
    e.emit(f"{res} = arith.select {bad}, {c1}, {c0} : i32")
    return res


# --------------------------------------------------------------------------
# Operator bodies (linalg surface)
# --------------------------------------------------------------------------


def _emit_matmul(e: Emitter, case: C.Case, st: Structural | None) -> tuple[str, str]:
    lhs_dims, rhs_dims, out_dims = case.dims["lhs"], case.dims["rhs"], case.dims["out"]
    lhs_r = concrete(lhs_dims, case.dyn, "lhs")
    rhs_r = concrete(rhs_dims, case.dyn, "rhs")
    out_r = concrete(out_dims, case.dyn, "out")
    lhs_t, rhs_t, out_t = (
        C.tensor_type(lhs_dims), C.tensor_type(rhs_dims), C.tensor_type(out_dims)
    )

    a = emit_formula_tensor(e, lhs_dims, lhs_r, offset=0, name="lhs")
    b = emit_formula_tensor(e, rhs_dims, rhs_r, offset=1000, name="rhs")

    init = emit_empty(e, out_dims, out_r)
    z = e.new("c")
    e.emit(f"{z} = arith.constant 0.0 : f32")
    filled = e.new("f")
    e.emit(f"{filled} = linalg.fill ins({z} : f32) outs({init} : {out_t}) -> {out_t}")

    res = e.new("mm")
    e.emit(
        f"{res} = linalg.matmul ins({a}, {b} : {lhs_t}, {rhs_t}) "
        f"outs({filled} : {out_t}) -> {out_t}"
    )
    return res, out_t


def _emit_relu(e: Emitter, case: C.Case, st: Structural | None) -> tuple[str, str]:
    inp_dims, out_dims = case.dims["inp"], case.dims["out"]
    inp_r = concrete(inp_dims, case.dyn, "inp")
    out_r = concrete(out_dims, case.dyn, "out")
    inp_t, out_t = C.tensor_type(inp_dims), C.tensor_type(out_dims)
    nd = len(inp_dims)

    x = emit_formula_tensor(e, inp_dims, inp_r, offset=0, name="inp")
    init = emit_empty(e, out_dims, out_r)

    dims = ",".join(f"d{i}" for i in range(nd))
    m = f"affine_map<({dims}) -> ({dims})>"
    res = e.new("y")
    e.emit(f"{res} = linalg.generic {{")
    e.indent += 1
    e.emit(f"indexing_maps = [{m}, {m}],")
    e.emit(f"iterator_types = [{iters_all_parallel(nd)}]")
    e.indent -= 1
    e.emit(f"}} ins({x} : {inp_t}) outs({init} : {out_t}) {{")
    e.indent += 1
    e.emit("^bb0(%v: f32, %o: f32):")
    z = e.new("c")
    e.emit(f"{z} = arith.constant 0.0 : f32")
    # `arith.maxf` was renamed in LLVM 21; `maximumf` propagates NaN like
    # numpy.maximum, which is what the reference oracle uses.
    mx = e.new("t")
    e.emit(f"{mx} = arith.maximumf %v, {z} : f32")
    e.emit(f"linalg.yield {mx} : f32")
    e.indent -= 1
    e.emit(f"}} -> {out_t}")
    return res, out_t


def _emit_transpose(e: Emitter, case: C.Case, st: Structural | None) -> tuple[str, str]:
    inp_dims, out_dims = case.dims["inp"], case.dims["out"]
    inp_r = concrete(inp_dims, case.dyn, "inp")
    out_r = concrete(out_dims, case.dyn, "out")
    inp_t, out_t = C.tensor_type(inp_dims), C.tensor_type(out_dims)

    x = emit_formula_tensor(e, inp_dims, inp_r, offset=0, name="inp")
    init = emit_empty(e, out_dims, out_r)
    res = e.new("tr")
    perm = ",".join(str(i) for i in reversed(range(len(inp_dims))))
    e.emit(
        f"{res} = linalg.transpose ins({x} : {inp_t}) "
        f"outs({init} : {out_t}) permutation = [{perm}]"
    )
    return res, out_t


def _emit_concat(e: Emitter, case: C.Case, st: Structural | None) -> tuple[str, str]:
    a_dims, b_dims, out_dims = case.dims["a"], case.dims["b"], case.dims["out"]
    a_r = concrete(a_dims, case.dyn, "a")
    b_r = concrete(b_dims, case.dyn, "b")
    out_r = concrete(out_dims, case.dyn, "out")
    a_t, b_t, out_t = (
        C.tensor_type(a_dims), C.tensor_type(b_dims), C.tensor_type(out_dims)
    )

    a = emit_formula_tensor(e, a_dims, a_r, offset=0, name="a")
    b = emit_formula_tensor(e, b_dims, b_r, offset=1000, name="b")
    res = e.new("cc")
    # LLVM 21 requires the function-type form: `dim(N) %a, %b : (T1, T2) -> T3`.
    e.emit(
        f"{res} = tensor.concat dim({C.CONCAT_AXIS}) {a}, {b} : "
        f"({a_t}, {b_t}) -> {out_t}"
    )
    return res, out_t


def _emit_softmax(e: Emitter, case: C.Case, st: Structural | None) -> tuple[str, str]:
    """exp(x - rowmax) / rowsum, normalized over the trailing axis."""
    inp_dims, out_dims = case.dims["inp"], case.dims["out"]
    inp_r = concrete(inp_dims, case.dyn, "inp")
    out_r = concrete(out_dims, case.dyn, "out")
    inp_t, out_t = C.tensor_type(inp_dims), C.tensor_type(out_dims)
    nd = len(inp_dims)

    x = emit_formula_tensor(e, inp_dims, inp_r, offset=0, name="inp")

    # Row tensors have shape (d0..d_{nd-2}, 1) so the trailing axis broadcasts.
    row_dims = tuple(inp_dims[:-1]) + (1,)
    row_r = tuple(inp_r[:-1]) + (1,)
    row_t = C.tensor_type(row_dims)

    dims = ",".join(f"d{i}" for i in range(nd))
    outdims = ",".join(f"d{i}" for i in range(nd - 1)) + ",0"
    inmap = f"affine_map<({dims}) -> ({dims})>"
    outmap = f"affine_map<({dims}) -> ({outdims})>"

    # --- row max ----------------------------------------------------------
    rinit = emit_empty(e, row_dims, row_r)
    ninf = e.new("c")
    e.emit(f"{ninf} = arith.constant 0xFF800000 : f32")  # -inf
    fmax = e.new("f")
    e.emit(f"{fmax} = linalg.fill ins({ninf} : f32) outs({rinit} : {row_t}) -> {row_t}")
    mx = e.new("mx")
    e.emit(f"{mx} = linalg.generic {{")
    e.indent += 1
    e.emit(f"indexing_maps = [{inmap}, {outmap}],")
    e.emit(f"iterator_types = [{iters_trailing_reduction(nd)}]")
    e.indent -= 1
    e.emit(f"}} ins({x} : {inp_t}) outs({fmax} : {row_t}) {{")
    e.indent += 1
    e.emit("^bb0(%v: f32, %acc: f32):")
    m2 = e.new("t")
    e.emit(f"{m2} = arith.maximumf %acc, %v : f32")
    e.emit(f"linalg.yield {m2} : f32")
    e.indent -= 1
    e.emit(f"}} -> {row_t}")

    # --- exp(x - rowmax): elementwise with a broadcast row ----------------
    # The outs operand must be a fresh tensor of the *input* type; reusing the
    # row-typed fill here would be a type error.
    einit = emit_empty(e, inp_dims, inp_r)
    e2 = e.new("e")
    e.emit(f"{e2} = linalg.generic {{")
    e.indent += 1
    e.emit(f"indexing_maps = [{inmap}, {outmap}, {inmap}],")
    e.emit(f"iterator_types = [{iters_all_parallel(nd)}]")
    e.indent -= 1
    e.emit(f"}} ins({x}, {mx} : {inp_t}, {row_t}) outs({einit} : {inp_t}) {{")
    e.indent += 1
    e.emit("^bb0(%v: f32, %m: f32, %o: f32):")
    d = e.new("t")
    e.emit(f"{d} = arith.subf %v, %m : f32")
    ex = e.new("t")
    e.emit(f"{ex} = math.exp {d} : f32")
    e.emit(f"linalg.yield {ex} : f32")
    e.indent -= 1
    e.emit(f"}} -> {inp_t}")

    # --- row sum of exp ---------------------------------------------------
    sinit = emit_empty(e, row_dims, row_r)
    z = e.new("c")
    e.emit(f"{z} = arith.constant 0.0 : f32")
    fsum = e.new("f")
    e.emit(f"{fsum} = linalg.fill ins({z} : f32) outs({sinit} : {row_t}) -> {row_t}")
    sm = e.new("sm")
    e.emit(f"{sm} = linalg.generic {{")
    e.indent += 1
    e.emit(f"indexing_maps = [{inmap}, {outmap}],")
    e.emit(f"iterator_types = [{iters_trailing_reduction(nd)}]")
    e.indent -= 1
    e.emit(f"}} ins({e2} : {inp_t}) outs({fsum} : {row_t}) {{")
    e.indent += 1
    e.emit("^bb0(%v: f32, %acc: f32):")
    a2 = e.new("t")
    e.emit(f"{a2} = arith.addf %acc, %v : f32")
    e.emit(f"linalg.yield {a2} : f32")
    e.indent -= 1
    e.emit(f"}} -> {row_t}")

    # --- divide -----------------------------------------------------------
    oinit = emit_empty(e, out_dims, out_r)
    res = e.new("y")
    e.emit(f"{res} = linalg.generic {{")
    e.indent += 1
    e.emit(f"indexing_maps = [{inmap}, {outmap}, {inmap}],")
    e.emit(f"iterator_types = [{iters_all_parallel(nd)}]")
    e.indent -= 1
    e.emit(f"}} ins({e2}, {sm} : {inp_t}, {row_t}) outs({oinit} : {out_t}) {{")
    e.indent += 1
    e.emit("^bb0(%v: f32, %s: f32, %o: f32):")
    dv = e.new("t")
    e.emit(f"{dv} = arith.divf %v, %s : f32")
    e.emit(f"linalg.yield {dv} : f32")
    e.indent -= 1
    e.emit(f"}} -> {out_t}")
    return res, out_t


def _emit_layer_norm(e: Emitter, case: C.Case, st: Structural | None) -> tuple[str, str]:
    """y = (x - mean)/sqrt(var + eps) * scale + bias over the trailing dim."""
    lhs_dims = case.dims["lhs"]
    scale_dims = case.dims["scale"]
    bias_dims = case.dims["bias"]
    out_dims = case.dims["out"]
    lhs_r = concrete(lhs_dims, case.dyn, "lhs")
    out_r = concrete(out_dims, case.dyn, "out")
    # scale/bias are 1-D over the trailing normalized dim. Their extents are
    # resolved from their *own* declared shape, so when M2.1/M2.3 declares a wrong
    # extent the operand genuinely has that many elements and the mismatch is
    # against what the linalg op expects -- the defect the spec describes.
    scale_r = concrete(scale_dims, case.dyn, "scale")
    bias_r = concrete(bias_dims, case.dyn, "bias")
    k = lhs_r[-1]
    lhs_t = C.tensor_type(lhs_dims)
    scale_t = C.tensor_type(scale_dims)
    bias_t = C.tensor_type(bias_dims)
    out_t = C.tensor_type(out_dims)
    nd = len(lhs_dims)

    x = emit_formula_tensor(e, lhs_dims, lhs_r, offset=0, name="lhs")
    sc = emit_formula_tensor(e, scale_dims, scale_r, offset=2000, name="scale")
    bi = emit_formula_tensor(e, bias_dims, bias_r, offset=3000, name="bias")

    row_dims = tuple(lhs_dims[:-1]) + (1,)
    row_r = tuple(lhs_r[:-1]) + (1,)
    row_t = C.tensor_type(row_dims)

    dims = ",".join(f"d{i}" for i in range(nd))
    outdims = ",".join(f"d{i}" for i in range(nd - 1)) + ",0"
    inmap = f"affine_map<({dims}) -> ({dims})>"
    outmap = f"affine_map<({dims}) -> ({outdims})>"
    kmap = f"affine_map<({dims}) -> (d{nd - 1})>"

    def zero_row() -> tuple[str, str]:
        init = emit_empty(e, row_dims, row_r)
        z = e.new("c")
        e.emit(f"{z} = arith.constant 0.0 : f32")
        f = e.new("f")
        e.emit(f"{f} = linalg.fill ins({z} : f32) outs({init} : {row_t}) -> {row_t}")
        return init, f

    ck = e.new("c")
    e.emit(f"{ck} = arith.constant {f32_lit(float(k))} : f32")

    # --- mean -------------------------------------------------------------
    _, f0 = zero_row()
    mean = e.new("mean")
    e.emit(f"{mean} = linalg.generic {{")
    e.indent += 1
    e.emit(f"indexing_maps = [{inmap}, {outmap}],")
    e.emit(f"iterator_types = [{iters_trailing_reduction(nd)}]")
    e.indent -= 1
    e.emit(f"}} ins({x} : {lhs_t}) outs({f0} : {row_t}) {{")
    e.indent += 1
    e.emit("^bb0(%v: f32, %acc: f32):")
    a2 = e.new("t")
    e.emit(f"{a2} = arith.addf %acc, %v : f32")
    e.emit(f"linalg.yield {a2} : f32")
    e.indent -= 1
    e.emit(f"}} -> {row_t}")

    # Divide the row sums by k. This is elementwise over the row tensor, so it
    # uses the identity map and all-parallel iterators.
    _, f1 = zero_row()
    mean_d = e.new("mean")
    e.emit(f"{mean_d} = linalg.generic {{")
    e.indent += 1
    e.emit(f"indexing_maps = [{inmap}, {inmap}],")
    e.emit(f"iterator_types = [{iters_all_parallel(nd)}]")
    e.indent -= 1
    e.emit(f"}} ins({mean} : {row_t}) outs({f1} : {row_t}) {{")
    e.indent += 1
    e.emit("^bb0(%v: f32, %o: f32):")
    dv = e.new("t")
    e.emit(f"{dv} = arith.divf %v, {ck} : f32")
    e.emit(f"linalg.yield {dv} : f32")
    e.indent -= 1
    e.emit(f"}} -> {row_t}")

    # --- var = mean((x - mean)^2) ----------------------------------------
    _, f2 = zero_row()
    sq = e.new("sq")
    e.emit(f"{sq} = linalg.generic {{")
    e.indent += 1
    e.emit(f"indexing_maps = [{inmap}, {outmap}, {outmap}],")
    e.emit(f"iterator_types = [{iters_trailing_reduction(nd)}]")
    e.indent -= 1
    e.emit(f"}} ins({x}, {mean_d} : {lhs_t}, {row_t}) outs({f2} : {row_t}) {{")
    e.indent += 1
    e.emit("^bb0(%v: f32, %m: f32, %acc: f32):")
    d = e.new("t")
    e.emit(f"{d} = arith.subf %v, %m : f32")
    m2 = e.new("t")
    e.emit(f"{m2} = arith.mulf {d}, {d} : f32")
    a3 = e.new("t")
    e.emit(f"{a3} = arith.addf %acc, {m2} : f32")
    e.emit(f"linalg.yield {a3} : f32")
    e.indent -= 1
    e.emit(f"}} -> {row_t}")

    _, f3 = zero_row()
    var = e.new("var")
    e.emit(f"{var} = linalg.generic {{")
    e.indent += 1
    e.emit(f"indexing_maps = [{inmap}, {inmap}],")
    e.emit(f"iterator_types = [{iters_all_parallel(nd)}]")
    e.indent -= 1
    e.emit(f"}} ins({sq} : {row_t}) outs({f3} : {row_t}) {{")
    e.indent += 1
    e.emit("^bb0(%v: f32, %o: f32):")
    dv2 = e.new("t")
    e.emit(f"{dv2} = arith.divf %v, {ck} : f32")
    e.emit(f"linalg.yield {dv2} : f32")
    e.indent -= 1
    e.emit(f"}} -> {row_t}")

    # --- normalize + scale + bias ----------------------------------------
    oinit = emit_empty(e, out_dims, out_r)
    eps = e.new("c")
    e.emit(f"{eps} = arith.constant 1.0e-5 : f32")
    res = e.new("y")
    e.emit(f"{res} = linalg.generic {{")
    e.indent += 1
    e.emit(f"indexing_maps = [{inmap}, {outmap}, {outmap}, {kmap}, {kmap}, {inmap}],")
    e.emit(f"iterator_types = [{iters_all_parallel(nd)}]")
    e.indent -= 1
    e.emit(
        f"}} ins({x}, {mean_d}, {var}, {sc}, {bi} : "
        f"{lhs_t}, {row_t}, {row_t}, {scale_t}, {bias_t}) "
        f"outs({oinit} : {out_t}) {{"
    )
    e.indent += 1
    e.emit("^bb0(%v: f32, %m: f32, %va: f32, %s: f32, %b: f32, %o: f32):")
    d1 = e.new("t")
    e.emit(f"{d1} = arith.subf %v, %m : f32")
    ve = e.new("t")
    e.emit(f"{ve} = arith.addf %va, {eps} : f32")
    sd = e.new("t")
    e.emit(f"{sd} = math.sqrt {ve} : f32")
    n1 = e.new("t")
    e.emit(f"{n1} = arith.divf {d1}, {sd} : f32")
    n2 = e.new("t")
    e.emit(f"{n2} = arith.mulf {n1}, %s : f32")
    n3 = e.new("t")
    e.emit(f"{n3} = arith.addf {n2}, %b : f32")
    e.emit(f"linalg.yield {n3} : f32")
    e.indent -= 1
    e.emit(f"}} -> {out_t}")
    return res, out_t


def _emit_elemwise_add(e: Emitter, case: C.Case,
                       st: Structural | None) -> tuple[str, str]:
    """`y = lhs + rhs`, elementwise, same shape on both operands.

    Composed from settings/elemwise_add.md case 1/11 (`f32 [I, J, K] lhs,
    f32 [I, J, K] rhs`). This is the level-2 M2 category (mutation-specs.md §5)
    and, unlike `matmul`, it *does* compose a binary elementwise op -- so it is
    the category where M2.3 (binary op on mismatched shapes) is genuinely
    expressible instead of `n/a`.
    """
    lhs_dims, rhs_dims = case.dims["lhs"], case.dims["rhs"]
    out_dims = case.dims["out"]
    lhs_r = concrete(lhs_dims, case.dyn, "lhs")
    rhs_r = concrete(rhs_dims, case.dyn, "rhs")
    out_r = concrete(out_dims, case.dyn, "out")
    lhs_t, rhs_t = C.tensor_type(lhs_dims), C.tensor_type(rhs_dims)
    out_t = C.tensor_type(out_dims)
    nd = len(lhs_dims)

    # Distinct offsets keep the two operands' values apart, matching `reference`.
    a = emit_formula_tensor(e, lhs_dims, lhs_r, offset=0, name="lhs")
    b = emit_formula_tensor(e, rhs_dims, rhs_r, offset=1000, name="rhs")
    init = emit_empty(e, out_dims, out_r)

    dims = ",".join(f"d{i}" for i in range(nd))
    m = f"affine_map<({dims}) -> ({dims})>"
    res = e.new("y")
    e.emit(f"{res} = linalg.generic {{")
    e.indent += 1
    e.emit(f"indexing_maps = [{m}, {m}, {m}],")
    e.emit(f"iterator_types = [{iters_all_parallel(nd)}]")
    e.indent -= 1
    e.emit(f"}} ins({a}, {b} : {lhs_t}, {rhs_t}) outs({init} : {out_t}) {{")
    e.indent += 1
    e.emit("^bb0(%x: f32, %y: f32, %o: f32):")
    s = e.new("t")
    e.emit(f"{s} = arith.addf %x, %y : f32")
    e.emit(f"linalg.yield {s} : f32")
    e.indent -= 1
    e.emit(f"}} -> {out_t}")
    return res, out_t


EMITTERS = {
    "matmul": _emit_matmul,
    "relu": _emit_relu,
    "transpose": _emit_transpose,
    "concat": _emit_concat,
    "softmax": _emit_softmax,
    "layer_normalization": _emit_layer_norm,
    "elemwise_add": _emit_elemwise_add,
}


# --------------------------------------------------------------------------
# Structural write defects (M2.5)
# --------------------------------------------------------------------------


def emit_structural_write(
    e: Emitter,
    computed: str,
    out_dims: tuple[int | None, ...],
    out_r: Sequence[int],
    st: Structural,
) -> str:
    """Re-express the output as a defective *write pattern*.

    M2.5 is not a type error -- the shapes all agree -- so it cannot be produced
    by perturbing dims. It is modelled the way the defect actually manifests in a
    tiled kernel: the output buffer is zero-initialized and only some tiles are
    written into it.

    * `partial-write`   -- the tail tile is omitted, so those elements stay zero.
    * `duplicate-write` -- the same tile is written twice, the second time at an
                           overlapping offset, so a row is clobbered by its
                           neighbour.

    Both are deterministic, which matters: leaving the tail genuinely
    uninitialized would make the oracle's verdict depend on whatever the allocator
    happened to hand back.
    """
    out_t = C.tensor_type(out_dims)
    nd = len(out_dims)
    axis = st.axis
    tile = st.tile

    # Zero-initialized destination buffer.
    init = emit_empty(e, out_dims, out_r)
    z = e.new("c")
    e.emit(f"{z} = arith.constant 0.0 : f32")
    dst = e.new("z")
    e.emit(f"{dst} = linalg.fill ins({z} : f32) outs({init} : {out_t}) -> {out_t}")

    # Static slice bounds: every axis takes the full extent except `axis`.
    sizes = list(out_r)
    sizes[axis] = tile
    offs = [0] * nd
    strides = [1] * nd
    sl = f"[{', '.join(str(o) for o in offs)}] " \
         f"[{', '.join(str(s) for s in sizes)}] " \
         f"[{', '.join(str(s) for s in strides)}]"
    tile_t = C.tensor_type(tuple(sizes))

    head = e.new("tile")
    e.emit(
        f"{head} = tensor.extract_slice {computed}{sl} : {out_t} to {tile_t}"
    )

    if st.kind == "partial-write":
        res = e.new("w")
        e.emit(
            f"{res} = tensor.insert_slice {head} into {dst}{sl} : "
            f"{tile_t} into {out_t}"
        )
        return res

    if st.kind == "duplicate-write":
        first = e.new("w")
        e.emit(
            f"{first} = tensor.insert_slice {head} into {dst}{sl} : "
            f"{tile_t} into {out_t}"
        )
        offs2 = list(offs)
        offs2[axis] = st.offset
        sl2 = f"[{', '.join(str(o) for o in offs2)}] " \
              f"[{', '.join(str(s) for s in sizes)}] " \
              f"[{', '.join(str(s) for s in strides)}]"
        res = e.new("w")
        e.emit(
            f"{res} = tensor.insert_slice {head} into {first}{sl2} : "
            f"{tile_t} into {out_t}"
        )
        return res

    raise KeyError(f"unhandled structural write kind {st.kind!r}")


# --------------------------------------------------------------------------
# Top-level module assembly
# --------------------------------------------------------------------------


def emit_kernel(
    case: C.Case,
    ref_case: C.Case | None = None,
    structural: Structural | None = None,
) -> str:
    """Compose a complete MLIR module for a (possibly mutated) case.

    `ref_case` is the *unmutated* case whose reference output is baked into the
    oracle. It defaults to `case` itself, so a clean kernel checks itself. A
    mutant must always be passed the clean case here: comparing a mutant against
    its own mutated reference would make every mutant look correct.
    """
    if case.category not in EMITTERS:
        raise KeyError(f"no emitter for category {case.category!r}")
    if ref_case is None:
        ref_case = case

    e = Emitter()
    e.indent = 0
    e.raw("func.func @main() -> i32 {")
    e.indent += 1

    # Everything emitted here is the kernel under audit, tagged loc("kernel").
    out_ssa, out_t = EMITTERS[case.category](e, case, structural)
    if structural is not None and structural.kind in ("partial-write",
                                                       "duplicate-write"):
        out_ssa = emit_structural_write(
            e, out_ssa, case.dims["out"],
            concrete(case.dims["out"], case.dyn, "out"), structural,
        )
    nd = len(case.dims["out"])
    ref = C.reference(ref_case)

    # The checksum oracle is harness machinery, not part of the kernel. Tagging it
    # separately is what lets `mlirbench.count_kernel_asserts` exclude its guards
    # from the S9 remainder count.
    with e.location("oracle"):
        res = emit_oracle(e, out_ssa, out_t, nd, ref)
    e.emit(f"return {res} : i32")
    e.indent -= 1
    e.raw("}")
    return e.text()
