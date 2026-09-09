#!/usr/bin/env python3
"""Kernel composition for the benchmark2 MLIR lanes.

Per plan §2.2 the benchmark *source* is not shared -- only the operator settings
(shapes, tensor contracts, reference semantics) in `benchmark2/settings/`. Each
lane composes its own kernels from those settings. This module is the
`mlir-linalg` / `mlir-low` composer.

Design decisions that matter for the numbers:

* **Deterministic inputs from a formula, not baked constants.** Input element
  `i` (flat index) is `((i * 37) % 13) - 6` as f32. The same formula is evaluated
  in numpy for the reference and emitted as `arith` ops in MLIR, so it scales to
  full-size inputs without emitting thousands of constants.
* **The oracle is a checksum, not a printout.** `@main` returns i32: `0` when the
  kernel's output matches the numpy reference, `1` when it does not. Two
  checksums (sum and sum-of-squares) are compared to keep accidental collisions
  unlikely. This single return value answers both the ref-check gate (§2.3) and
  the §7 manifest oracle (`corrupts` vs `noop`) at any input size.
* **Small vs full follow the lane convention** pinned in `manifest.md` §5.2:
  rank and the static-vs-dynamic *pattern* are preserved exactly; only the
  extents shrink.

Mutations are applied by perturbing the composition parameters, never by editing
emitted text -- a `sed`-style mutation can silently fail to match and leave a
plain type error that looks like a detection.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

# --------------------------------------------------------------------------
# Input formula (shared by the MLIR emitter and the numpy reference)
# --------------------------------------------------------------------------

_MUL, _MOD, _SUB = 37, 13, 6


def input_values(n: int, offset: int = 0) -> np.ndarray:
    """Deterministic f32 input of length `n`, matching the emitted MLIR.

    `offset` shifts the flat index so a second operand gets a different (but
    still reproducible) value stream without changing the formula.
    """
    i = np.arange(offset, offset + n, dtype=np.int64)
    return (((i * _MUL) % _MOD) - _SUB).astype(np.float32)


# Position weights for the third (order-sensitive) checksum. Deliberately a
# different multiplier/modulus than the input formula above: reusing 37/13 would
# make the weight stream correlated with the value stream and weaken the check.
# _W_MOD > 48 (the largest small-size output) keeps every weight in the level-1
# gate distinct; _W_SUB centres them so the weighted sum stays near zero instead
# of drifting to a large positive bias.
_W_MUL, _W_MOD, _W_SUB = 41, 127, 63


def position_weights(n: int) -> np.ndarray:
    """Weight for flat position `i` is `((i * _W_MUL) % _W_MOD) - _W_SUB`.

    Computed in i64 then widened to f64, matching the emitted MLIR exactly: the
    kernel derives the weight from an `index` value with `arith.remsi`, so doing
    the same here is what keeps the numpy oracle and the in-kernel oracle in
    bit-for-bit agreement.
    """
    i = np.arange(n, dtype=np.int64)
    return (((i * _W_MUL) % _W_MOD) - _W_SUB).astype(np.float64)


# Relative tolerance constants for the checksum oracle. Both chosen from
# measurement, not guessed -- see `tolerance`.
_TOL_REL = 1.0e-5
_TOL_DRIFT = 0.5
_EPS_F32 = 1.1920929e-7


def tolerance(nel: int, ref_value: float) -> float:
    """Absolute tolerance for comparing one computed checksum against `ref_value`.

    Two terms, because the kernel's reduction accumulates error in two different
    regimes:

    `_TOL_REL * sqrt(nel) * max(1, |ref_value|)`
        The random-walk term. When the summands cancel, rounding error grows like
        a random walk, i.e. as `eps * sqrt(nel)`. This is the term that governs
        every small-size check, and it is what makes the comparison *relative*:
        the previous absolute form (`1e-2 * sqrt(nel)`) was ~1000x too loose and
        let a real defect through -- softmax/static with a dropped boundary mask
        shifts `sum-of-squares` by 1.35e-2, which sat under the 2.83e-2 absolute
        budget and was scored `noop`, a false success. Measured on the same
        kernels, the largest clean discrepancy is 8.2e-7 *relative*, so the
        defect sits ~4500x above the noise floor.

    `_TOL_DRIFT * nel * eps_f32 * max(1, |ref_value|)`
        The monotone-drift term, and it is not optional at full size. A checksum
        whose summands are all the same sign -- `sum-of-squares` always is --
        does not random-walk: the accumulator grows monotonically, and once it
        passes 2^24 f32 stops representing integers exactly, so every subsequent
        addition rounds at `eps * accumulator`. Error then grows *linearly* in
        `nel`. Measured: `elemwise_add` at full size (12,582,912 elements)
        accumulates 3.2e7 of drift on a `sum-of-squares` of 4.3e8 -- a 7.5e-2
        relative error, which the sqrt term alone budgets at 1.5e7 and so fails a
        *clean* kernel. `_TOL_DRIFT = 0.5` gives that measurement ~10x margin.

        This term is negligible wherever it does not matter: at nel=8 it is
        8.4e-7 against the sqrt term's 5.0e-5, so small-size sensitivity is
        unchanged.

    Consequence worth stating plainly: at full size the `sum-of-squares` check on
    a large-magnitude category is drift-limited and nearly blind. Detection there
    rests on the `sum` and position-weighted checks, whose summands cancel and so
    stay tight (measured drift 0.0 on every full-size category). The three checks
    are OR-ed for exactly this reason.

    The `max(1, ...)` floor keeps the tolerance from collapsing to zero for a
    checksum that happens to sum to ~0 through cancellation, which would
    otherwise fail a clean kernel on rounding alone.
    """
    scale = max(1.0, abs(float(ref_value)))
    n = float(max(1, int(nel)))
    walk = _TOL_REL * float(np.sqrt(n)) * scale
    drift = _TOL_DRIFT * n * _EPS_F32 * scale
    return walk + drift


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One composed kernel instance.

    `dims` maps operand name -> tuple of extents. An extent of `None` means
    *dynamic*; `dyn` binds each dynamic extent to a concrete value at the entry
    (per manifest §5.2 rule 3).
    """

    category: str
    case_id: str
    size: str  # "small" | "full"
    dims: dict[str, tuple[int | None, ...]]
    dyn: dict[str, int] = field(default_factory=dict)

    def is_dynamic(self) -> bool:
        return any(d is None for ds in self.dims.values() for d in ds)

    def shape(self, name: str) -> tuple[int, ...]:
        """Concrete shape, with dynamic extents bound to their `dyn` values."""
        out = []
        di = 0
        for d in self.dims[name]:
            if d is None:
                out.append(self.dyn[f"{name}.{di}"])
            else:
                out.append(d)
            di += 1
        return tuple(out)

    def numel(self, name: str) -> int:
        return int(np.prod(self.shape(name)))


# --------------------------------------------------------------------------
# Mutations (M1 element-access, M2 shape-compatibility)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Mutation:
    """A defect spec from `specs/mutation-specs.md`, applied to a Case.

    `spec` is the 1-based spec number within the class. `variant` distinguishes
    sub-defects inside one spec (spec M2.5 names two: an omitted tail tile *and*
    an overlapping duplicate write), so each gets its own mutant id and its own
    record rather than being counted twice for the same thing.

    Two identities matter and they are NOT the same number:

    * `spec_id` -- the **injection** identity. This is what S1's `n_injected`
      counts, because mutation-specs.md §2 numbers M2 as 5 specs and §0 fixes
      N = 40 per class at *spec* level. Spec 5 counts once no matter how many
      variants are emitted.
    * `mutant_id` -- the **record** identity. One per emitted kernel, so spec 5's
      two variants produce two records and two measurements.
    """

    klass: str  # "M1" | "M2" | "M3"
    spec: int
    name: str
    paper_category: str  # dim-mismatch | oob | wrong-shape | stride
    target: str = ""  # operand name the perturbation applies to
    delta: int = 0
    detail: str = ""
    variant: str = ""

    @property
    def spec_id(self) -> str:
        """Injection identity: the spec number, ignoring variants."""
        return f"{self.klass}.{self.spec}"

    @property
    def mutant_id(self) -> str:
        """Record identity: one per emitted kernel."""
        return f"{self.klass}.{self.spec}" + (f".{self.variant}" if self.variant else "")


# M2 -- shape-compatibility (mlir-linalg's only assigned class).
# Specs are from mutation-specs.md §2, which numbers M2 as FIVE specs. Spec 5
# names two distinct defects ("partial write (omitted tail tile) / duplicate write
# (overlapping tile)"), so it is expanded into two variants: two records, but ONE
# injected spec for the purposes of N = 40 (§0). Both variants are real defects --
# each lands in the `never`/`corrupts` row, the silent-bug residue -- so neither is
# dropped; they are two measurements of one injection.
M2_SPECS: list[Mutation] = [
    Mutation("M2", 1, "secondary-operand-wrong-extent", "dim-mismatch",
             detail="secondary operand (rhs / b / scale) has the wrong leading extent"),
    Mutation("M2", 2, "ins-outs-extent-disagreement", "dim-mismatch",
             detail="output extent disagrees with what the inputs imply "
                    "(the DMA source/destination analogue)"),
    Mutation("M2", 3, "binary-op-mismatched-shapes", "dim-mismatch",
             detail="the two operands of a binary op disagree in extent"),
    Mutation("M2", 4, "output-wrong-leading-extent", "wrong-shape",
             detail="output tensor declared with a wrong leading extent"),
    Mutation("M2", 5, "partial-write", "wrong-shape", variant="partial",
             detail="omitted tail tile: only a prefix of the output is written"),
    Mutation("M2", 5, "duplicate-write", "wrong-shape", variant="duplicate",
             detail="overlapping tile written twice at a shifted offset"),
]

# The five numbered M2 specs, in order, with variants collapsed. This is the list
# to iterate when counting injections (N); iterate M2_SPECS when emitting records.
M2_SPEC_IDS: list[str] = ["M2.1", "M2.2", "M2.3", "M2.4", "M2.5"]

# mutation-specs.md §5 splits M2 coverage into a level-1 minimal set and level-2
# additions. The split is carried into each mutant record's `level` field, whose
# schema enum is exactly {"1","2"}. Kept here (not in a validator) so the record
# writer and the census cannot drift apart.
LEVEL1_M2_CATS: list[str] = ["layer_normalization", "matmul", "concat"]
LEVEL2_M2_CATS: list[str] = ["elemwise_add", "softmax"]
M2_CATS: list[str] = LEVEL1_M2_CATS + LEVEL2_M2_CATS
LEVEL_OF: dict[str, str] = {c: "1" for c in LEVEL1_M2_CATS}
LEVEL_OF.update({c: "2" for c in LEVEL2_M2_CATS})

# §0's per-class injection target. Level-1 alone is 30 (under target); adding both
# §5 level-2 M2 categories gives 5 specs x 5 cats x 2 shapes = 50. The +10
# overshoot is deliberate and owner-approved -- see mlir-shared/README.md. Never
# trimmed silently to make the arithmetic land on 40.
N_TARGET_PER_CLASS = 40

# M1 -- element-access (mlir-low's only assigned class).
# Specs are from mutation-specs.md §1.
M1_SPECS: list[Mutation] = [
    Mutation("M1", 1, "dropped-boundary-mask", "oob",
             detail="boundary guard omitted on the indexed access"),
    Mutation("M1", 2, "off-by-one", "oob",
             detail="index off by one past the end"),
    Mutation("M1", 3, "negative-index", "oob",
             detail="index goes negative"),
    Mutation("M1", 4, "transposed-stride", "stride",
             detail="transposed / non-contiguous stride"),
    Mutation("M1", 5, "offset-view-overrun", "oob",
             detail="base + offset overruns the tensor"),
    Mutation("M1", 6, "zero-stride-empty-range", "stride",
             detail="zero stride / empty iteration range"),
]


# --------------------------------------------------------------------------
# MLIR type helpers
# --------------------------------------------------------------------------


def _dim(d: int | None) -> str:
    return "?" if d is None else str(d)


def tensor_type(dims: tuple[int | None, ...], dtype: str = "f32") -> str:
    if not dims:
        return f"tensor<{dtype}>"
    return "tensor<" + "x".join(_dim(d) for d in dims) + f"x{dtype}>"


def memref_type(dims: tuple[int | None, ...], dtype: str = "f32") -> str:
    if not dims:
        return f"memref<{dtype}>"
    return "memref<" + "x".join(_dim(d) for d in dims) + f"x{dtype}>"


# --------------------------------------------------------------------------
# Small-input convention (manifest §5.2)
# --------------------------------------------------------------------------
#
# Rank and the static/dynamic pattern are preserved; extents shrink to the
# smallest values that keep the op semantics legal. Contraction dims stay equal
# on both operands.

SMALL_DIMS: dict[str, dict[str, tuple[int | None, ...]]] = {
    # --- mlir-linalg (M2) ---
    "matmul": {"lhs": (2, 3), "rhs": (3, 4), "out": (2, 4)},
    "concat": {"a": (2, 2, 2, 2), "b": (2, 3, 2, 2), "out": (2, 5, 2, 2)},
    "layer_normalization": {
        "lhs": (2, 2, 4), "scale": (4,), "bias": (4,), "out": (2, 2, 4)
    },
    # level-2 M2 addition (mutation-specs.md §5). Rank-3 same-shape variant, per
    # settings case 1 (`1_bert_32x512x768`) / case 11 (`11_dynamic_32xSx768`).
    "elemwise_add": {"lhs": (2, 2, 3), "rhs": (2, 2, 3), "out": (2, 2, 3)},
    # --- mlir-low (M1) ---
    "relu": {"inp": (2, 3), "out": (2, 3)},
    "softmax": {"inp": (2, 4), "out": (2, 4)},
    "transpose": {"inp": (2, 3), "out": (3, 2)},
}

# Full-size extents, taken from the first concrete case in each settings file.
FULL_DIMS: dict[str, dict[str, tuple[int | None, ...]]] = {
    "matmul": {"lhs": (128, 1280), "rhs": (1280, 256), "out": (128, 256)},
    "concat": {"a": (4, 64, 8, 8), "b": (4, 96, 8, 8), "out": (4, 160, 8, 8)},
    "layer_normalization": {
        "lhs": (32, 64, 128), "scale": (128,), "bias": (128,), "out": (32, 64, 128)
    },
    # settings/elemwise_add.md case 11: `11_dynamic_32xSx768`.
    "elemwise_add": {"lhs": (32, 512, 768), "rhs": (32, 512, 768),
                     "out": (32, 512, 768)},
    "relu": {"inp": (32, 512, 8, 8), "out": (32, 512, 8, 8)},
    "softmax": {"inp": (16, 512, 8, 8), "out": (16, 512, 8, 8)},
    "transpose": {"inp": (32, 64), "out": (64, 32)},
}

# Which operand(s) carry a dynamic extent, per category (mirrors the settings'
# `N`, `H`, `W`, `seq_len` symbolic dims). A category may need MORE THAN ONE
# slot: `elemwise_add`'s signature is `f32 [I, N, K] lhs, f32 [I, N, K] rhs`, so
# the symbolic `N` appears on both operands and must be made dynamic on both --
# marking only `lhs` would compose a kernel whose operands disagree, i.e. it
# would inject an M2 defect into what is supposed to be the clean case.
DYNAMIC_SLOT: dict[str, tuple[tuple[str, int], ...]] = {
    "matmul": (("rhs", 1),),
    "concat": (("a", 1),),
    "layer_normalization": (("lhs", 0),),
    "elemwise_add": (("lhs", 1), ("rhs", 1)),
    "relu": (("inp", 0),),
    "softmax": (("inp", 0),),
    "transpose": (("inp", 0),),
}

_DYN_VALUE = 3

# Concat axis, per the settings signature `ele_concat_J`.
CONCAT_AXIS = 1


def _derive_output(category: str, dims: dict[str, tuple[int | None, ...]],
                   dyn: dict[str, int]) -> None:
    """Recompute the output shape from the operand shapes (in place).

    Dynamic extents must propagate: `matmul`'s output follows `rhs`'s trailing
    dim, `concat`'s output axis is the *sum* of the two operand axes,
    `transpose` swaps. Getting this wrong silently produces a shape mismatch
    that looks like a detection.
    """
    def concrete(name: str, axis: int) -> int:
        d = dims[name][axis]
        return dyn[f"{name}.{axis}"] if d is None else d

    if category == "matmul":
        dims["out"] = (dims["lhs"][0], dims["rhs"][1])
        if dims["rhs"][1] is None:
            dyn["out.1"] = dyn["rhs.1"]
    elif category == "concat":
        ax = CONCAT_AXIS
        total = concrete("a", ax) + concrete("b", ax)
        out = list(dims["out"])
        out[ax] = None if (dims["a"][ax] is None or dims["b"][ax] is None) else total
        dims["out"] = tuple(out)
        if out[ax] is None:
            dyn["out.%d" % ax] = total
    elif category == "transpose":
        dims["out"] = tuple(reversed(dims["inp"]))
        for axis, d in enumerate(dims["inp"]):
            if d is None:
                dyn[f"out.{len(dims['inp']) - 1 - axis}"] = dyn[f"inp.{axis}"]
    elif category in ("relu", "softmax", "layer_normalization", "elemwise_add"):
        src = "inp" if "inp" in dims else "lhs"
        dims["out"] = dims[src]
        for axis, d in enumerate(dims[src]):
            if d is None:
                dyn[f"out.{axis}"] = dyn[f"{src}.{axis}"]


def make_case(category: str, size: str = "small", dynamic: bool = False) -> Case:
    """Build a Case for a category at the requested size and shape mode."""
    table = SMALL_DIMS if size == "small" else FULL_DIMS
    if category not in table:
        raise KeyError(f"no composed dims for category {category!r}")
    dims = dict(table[category])

    dyn: dict[str, int] = {}
    if dynamic:
        if category not in DYNAMIC_SLOT:
            raise KeyError(f"no dynamic slot for {category!r}")
        for op, axis in DYNAMIC_SLOT[category]:
            cur = list(dims[op])
            if axis >= len(cur):
                raise ValueError(
                    f"{category}: dynamic axis {axis} out of range for {op!r}"
                )
            cur[axis] = None
            dims[op] = tuple(cur)
            # All slots of one category share the symbolic dim (e.g. `N` on both
            # `lhs` and `rhs` of elemwise_add), so they bind to one value.
            dyn[f"{op}.{axis}"] = _DYN_VALUE

    _derive_output(category, dims, dyn)

    return Case(
        category=category,
        case_id=f"{category}_{'dyn' if dynamic else 'static'}",
        size=size,
        dims=dims,
        dyn=dyn,
    )


# --------------------------------------------------------------------------
# Reference semantics (numpy) -- the ground truth for the §7 oracle
# --------------------------------------------------------------------------


def reference(case: Case) -> np.ndarray:
    """Compute the reference output for a case, per its settings semantics."""
    cat = case.category
    if cat == "matmul":
        a = input_values(case.numel("lhs")).reshape(case.shape("lhs"))
        b = input_values(case.numel("rhs"), offset=1000).reshape(case.shape("rhs"))
        return (a @ b).astype(np.float32)
    if cat == "relu":
        x = input_values(case.numel("inp")).reshape(case.shape("inp"))
        return np.maximum(x, 0.0).astype(np.float32)
    if cat == "softmax":
        x = input_values(case.numel("inp")).reshape(case.shape("inp"))
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
    if cat == "transpose":
        x = input_values(case.numel("inp")).reshape(case.shape("inp"))
        return np.transpose(x).astype(np.float32)
    if cat == "concat":
        a = input_values(case.numel("a")).reshape(case.shape("a"))
        b = input_values(case.numel("b"), offset=1000).reshape(case.shape("b"))
        return np.concatenate([a, b], axis=CONCAT_AXIS).astype(np.float32)
    if cat == "elemwise_add":
        # settings: `y = lhs + rhs; elementwise with broadcast`. The composed
        # variant is the same-shape one (settings cases 1/11), so no broadcasting
        # is exercised here; `rhs` takes offset 1000 to match `concat`'s second
        # operand and keep the two operands' values distinct.
        a = input_values(case.numel("lhs")).reshape(case.shape("lhs"))
        b = input_values(case.numel("rhs"), offset=1000).reshape(case.shape("rhs"))
        return (a + b).astype(np.float32)
    if cat == "layer_normalization":
        x = input_values(case.numel("lhs")).reshape(case.shape("lhs"))
        k = case.shape("scale")[0]
        scale = input_values(k, offset=2000).reshape(case.shape("scale"))
        bias = input_values(k, offset=3000).reshape(case.shape("bias"))
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        eps = np.float32(1e-5)
        y = (x - mean) / np.sqrt(var + eps)
        # scale/bias broadcast over the trailing normalized dim(s).
        y = y * scale.reshape((1,) * (y.ndim - scale.ndim) + scale.shape)
        y = y + bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)
        return y.astype(np.float32)
    raise KeyError(f"no reference semantics for category {cat!r}")


def checksums(arr: np.ndarray) -> tuple[float, float, float]:
    """(sum, sum-of-squares, position-weighted sum) as f32 -- baked into the kernel.

    The third checksum is not redundant. `sum` and `sum-of-squares` are both
    *permutation-invariant*, so an M1 defect that reorders values -- off-by-one,
    negative index, transposed stride -- leaves both untouched and the mutant is
    scored `noop`, a false success that specs §7.1 requires be discarded.
    Measured on
    relu/static: clean `[[0,5,3],[1,0,0]]` and the M1.2 mutant `[[5,3,1],[0,0,0]]`
    both give sum=9, sumsq=35. Weighting each element by its position breaks the
    symmetry (9 vs -37).

    `_W_MOD` is chosen larger than any small-size output (max 48 elements, see
    `_validate.py`), so every weight in the level-1 gate is distinct and the
    weighted sum is injective on value placements: no permutation can evade it.
    """
    a = arr.astype(np.float64)
    # Ravel in C order: the kernel derives its flat index row-major from the
    # induction variables, so this is the same position numbering.
    flat = a.ravel()
    w = position_weights(int(flat.size))
    return (
        float(np.float32(a.sum())),
        float(np.float32((a * a).sum())),
        float(np.float32((flat * w).sum())),
    )
