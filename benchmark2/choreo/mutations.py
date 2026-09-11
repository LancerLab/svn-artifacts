#!/usr/bin/env python3
"""Mutation operator library for the choreo lane (E1).

Implements specs/mutation-specs-v2.md §1-§4 as concrete source transforms on the
choreo reference surface, and carries the v2.1 **check-path classification**
(§9.6) and **applicability rule** (§9.5.0) on every operator.

Workflow: specs/expansion-workflow.md. GATE 1 (screen by path class) and GATE 2
(operator resolves against the base kernel) are both enforced here and by
`gen_mutants.py --report`.

Design note: transforms are keyed by (class, category, case), not by class
alone. The choreo suite is hand-written per case, so `scale.at(k)` exists in
layer_norm but not in transpose. A class-level table therefore silently skips
most (transform, case) pairs. Keying per case means every transform in this file
names source text that actually exists in the kernel it targets, and a skip is a
real defect in this file rather than an expected mismatch.

Ground rules (specs v2.1 §0, §6, §7):
  * 5 classes: M1 element-access, M2 shape-compatibility, M3 hardware-constraint,
    M4 iteration-validity, L launch-status.
  * Every spec names a **path class** (§9.6): P1 assessed, P2 hard-error,
    P3 unchecked, P4 warning-only. L specs are `path_class="L"` — they are
    attribution-only and never enter an admissible denominator (§4).
  * A transform that does not change the source is a `noop` and is rejected.
  * Counts are reached by enumerating defect *magnitudes* within a spec (e.g.
    K % 16 with residues 2/4/6/8/12), never by inventing defects outside §1-§4.
  * **Budget follows the path** (§9.6.1): P1 cells get the `-rtc` curve sweep;
    P3/P4 cells get ONE injection per (spec x surface) cell; P2 and
    noop-by-construction specs get ZERO injections and are recorded as N/A.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Base cases. Case 1 is the static-shape instance; the second is dynamic. E1
# needs both because the compile/runtime boundary follows the static-vs-dynamic
# shape boundary (specs §6), so a class mutating only static kernels could not
# populate the runtime column at all.
# ---------------------------------------------------------------------------
BASE_CASES = {
    "layer_normalization": [
        "1_bert_32x512x768_768_768",
        "3_attention_32xNx512x64_64_64",
    ],
    "softmax": [
        "1_bert_32x512x768_32x512x768",
        "11_dynamic_32xSx768_32xSx768",
    ],
    "relu": [
        "1_bert_32x512x768_32x512x768",
        "11_dynamic_32xSx768_32xSx768",
    ],
    "transpose": [
        "1_bert_32x512x768_32x768x512",
        "11_dynamic_32xSx768_32x768xS",
    ],
    "matmul": [
        "1_bert_32x512x768_768x768_32x512x768",
        "11_dynamic_32xSx768_768x768_32xSx768",
    ],
    "concat": [
        "1_bert_32x512x768_32x512x768_32x512x1536",
        "11_dynamic_32xS1x768_32xS2x768_32xS1pS2x768",
    ],
    "conv2d": [
        "1_attention_32x512xHxW_1024x512x1x1_32x1024xHxW_1_0_1",
        "11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1",
    ],
}

# Minimal coverage set (specs §5).
#
# v2.1 widened M2's minimal set with conv2d (the only category with an explicit
# `.pad` and `span_as` surface, needed by M2.15/M2.16), plus relu and softmax
# (the DMA-to-buffer and MSB-broadcast surfaces needed by M2.11/M2.21). Those
# two move OUT of M2's level-2 order below, so nothing is counted twice.
#
# M4 and L are v2.1. M4 is generated inside the SAME four kernels as M1-M3
# because an iteration-space defect is only meaningful in a body that already
# carries obligations; L rides on conv2d because that is where the block and
# shared extents are explicit.
MINIMAL_SET = {
    "M1": ["layer_normalization", "softmax", "relu", "transpose"],
    "M2": ["layer_normalization", "matmul", "concat", "conv2d", "relu",
           "softmax"],
    "M3": ["matmul", "conv2d"],
    "M4": ["conv2d", "relu", "transpose"],
    "L": ["conv2d"],
}

# Level-2 widening order (specs §5), applied only after level-1 is green.
LEVEL2_SET = {
    "M1": ["max_pool2d", "conv2d", "embedding", "batch_norm"],
    "M2": ["elemwise_add"],
    "M3": ["batch_norm"],
    "M4": [],
    "L": [],
}


# ===========================================================================
# SPEC_REGISTRY -- the spec of record, in code (specs v2.1 §1-§4, §9.6).
#
# This is GATE 1. Every mutation spec the suite claims to cover has one entry
# here, carrying the three things v2.1 added to the vocabulary:
#
#   path          P1 assessed | P2 hard-error | P3 unchecked | P4 warning-only
#                 ("L" = launch-status, attribution-only, never admissible)
#   admissible    may this spec contribute to the admissible denominator?
#                 False for noop-by-construction specs and for every L spec.
#   prohibition   if not applicable, WHY (specs §9.5.0):
#                   absent       the suite cannot express the corrupted state
#                                (no source surface); include the missing
#                                surface in `note` so the gap is actionable
#                   derived      the prohibition is not stated, only derived
#                   repaired     the model FORBIDS the state, so the compiler
#                                refuses it and there is no test to run (P2)
#                   harness-owned the check lives in the harness, not the
#                                compiler (the E2 obligation suite)
#                   observation  attribution-only (the launch-status class):
#                                the mutant is generated but its outcome can
#                                only ever be "rejected launch", so it enters
#                                no admissible denominator
#
# TWO ORTHOGONAL AXES. `status` records whether an operator EXISTS; `admissible`
# records whether the spec may enter the denominator. They are deliberately not
# merged: M1.6/M3.6/M4.4/L1/L2/L4 ARE generated (their non-detection is the
# point) yet stay out of the denominator, so collapsing the two would either
# lose the controls or inflate the denominator.
#
# status is the honest state of the implementation:
#   "implemented" the spec has >=1 operator in this file
#   "pending"     screened and admissible, operator not written yet  <- the
#                 only state that represents open work
#
# `gen_mutants.py --report` cross-checks this registry against the operators
# that actually loaded, so a "pending" spec is visible rather than silent, and
# an operator that claims a spec absent from the registry is an error.
# ===========================================================================

# Noop-by-construction specs: retained in the corpus as named controls, excluded
# from the admissible denominator (specs §1 M1.6, §9.1 M4.4).
NOOP_BY_CONSTRUCTION = ("M1.6", "M4.4")


def _spec(cls, path, desc, admissible=True, prohibition="", status="pending",
          note=""):
    return {"cls": cls, "path": path, "admissible": admissible,
            "prohibition": prohibition, "status": status, "desc": desc,
            "note": note}


SPEC_REGISTRY = {
    # ---- M1 element-access (21 specs, specs §1) -------------------------
    "M1.1": _spec("M1", "P1", "dropped boundary mask", status="implemented"),
    "M1.2": _spec("M1", "P1", "p#n off-by-one", status="implemented"),
    "M1.3": _spec("M1", "P1", "negative index", status="implemented"),
    "M1.4": _spec("M1", "P1", "transposed / non-contiguous stride",
                  status="implemented"),
    "M1.5": _spec("M1", "P1", "offset-view overrun", status="implemented"),
    "M1.6": _spec("M1", "P1", "zero-stride / empty range",
                  admissible=False, prohibition="absent",
                  status="implemented",
                  note="noop by construction -- there is no prohibition to "
                       "enforce, because the range is correctly empty. "
                       "Retained for the miss audit, excluded from the "
                       "admissible denominator"),
    "M1.7": _spec("M1", "P1", "reversed loop bound (upper < lower) -> overrun "
                              "instead of empty"),
    "M1.8": _spec("M1", "P1", "stride scaling (stride x 2)"),
    "M1.9": _spec("M1", "P1", "base offset applied WITHOUT shrinking the extent"),
    "M1.10": _spec("M1", "P1", "tile-boundary rounding (floor vs ceil on the "
                               "last tile)"),
    "M1.11": _spec("M1", "P1", "read-after-write aliasing overlap"),
    "M1.12": _spec("M1", "P1", "wrong loop variable for a dimension "
                               "(broadcast index reuse)"),
    "M1.13": _spec("M1", "P1", "symbolic-bound overrun (index beyond a bound "
                               "over a runtime parameter)"),
    "M1.14": _spec("M1", "P1", "chunkat tile-coordinate over/underflow while "
                               "the element index stays in bounds",
                  note="load-bearing: the only spec targeting a check the "
                       "compiler wrote and then disabled (272 chunkat "
                       "obligations, 0 enabled at -rtc=entry)"),
    "M1.15": _spec("M1", "P3", "dimof index >= rank, non-constant index",
                  admissible=False, prohibition="absent",
                  note="gap spec: the dimof-rank mechanism neither exists in "
                       "the source suite nor, per specs §1.1, emits any "
                       "runtime obligation (0/974). MISSING SURFACE: a case "
                       "using `dimof` with a runtime index"),
    "M1.16": _spec("M1", "P3", "select factor out of range (f >= count | f < 0)",
                  admissible=False, prohibition="absent",
                  note="semacheck.cpp:2032-2038, the no-static-factor path is "
                       "unassessed (defect F10) -- but no case in the suite "
                       "uses `select`, so the rule is never reached. "
                       "MISSING SURFACE: a case with a runtime select factor"),
    "M1.17": _spec("M1", "P3", "5th-index access on a rank-5 view",
                  admissible=False, prohibition="absent",
                  note="MISSING SURFACE: the suite's maximum rank is 4, so a "
                       "rank-5 view cannot be constructed"),
    "M1.18": _spec("M1", "P1", "index in range by `interval` but out of range "
                               "by `canonical` (or vice versa)",
                  admissible=False, prohibition="absent",
                  note="specs v2.1 states the intent but gives no "
                       "realization, and the two mechanisms are internal "
                       "(shapeinfer). MISSING SURFACE: a case whose index map "
                       "makes `interval` and `canonical` disagree -- must be "
                       "identified before this spec can be written"),
    "M1.19": _spec("M1", "P3", "index-carrier overflow (realization b, "
                               "narrow-carrier)", status="implemented",
                  note="adopted from §9.5.1; the (int) cast at "
                       "cute_codegen.cpp:1757 is also a latent shipped defect"),
    "M1.20": _spec("M1", "P3", "view / subspan offset, stride, rank arity "
                               "(_StaticFail_-only; symbols neither refused "
                               "nor checked)", status="implemented",
                  note="P3 for a symbolic offset, P2 (repaired) for a static "
                       "one -- the split must be explicit or the noop rate is "
                       "an artefact"),
    "M1.21": _spec("M1", "P1", "tileat/at index vs the tiled extent; step/"
                               "stride tail on a non-divisible extent",
                  status="implemented",
                  note="P1 per-dim, P2-P3 for the composition"),

    # ---- M2 shape-compatibility (20 specs, specs §2) --------------------
    "M2.1": _spec("M2", "P1", "wrong leading extent for a secondary operand",
                  status="implemented"),
    "M2.2": _spec("M2", "P1", "DMA src/dst extent disagreement",
                  status="implemented"),
    "M2.3": _spec("M2", "P1", "binary op on mismatched shapes",
                  status="implemented"),
    "M2.4": _spec("M2", "P1", "wrong output leading extent",
                  status="implemented"),
    "M2.5": _spec("M2", "P1", "partial write (omitted tail tile) / duplicate "
                              "write (overlapping tile)", status="implemented"),
    "M2.6": _spec("M2", "P1", "two extents transposed"),
    "M2.7": _spec("M2", "P1", "reduced-rank view (a dimension dropped)"),
    "M2.8": _spec("M2", "P1", "broadcast extent set to 1 instead of N"),
    "M2.9": _spec("M2", "P1", "batch/group dimension swapped"),
    "M2.10": _spec("M2", "P1", "transpose permutation on a SQUARE operand "
                               "(extents equal, memory order changes)",
                   note="view/metadata family -- best-attested class in both "
                        "empirical corpora; report as its own sub-table"),
    "M2.11": _spec("M2", "P1", "DMA to-buffer element-count undersize on a "
                               "LogicalEqual path"),
    "M2.12": _spec("M2", "P2", "rank mismatch through .pad (overlap still "
                               "satisfies f + pad == t)",
                   admissible=False, prohibition="repaired",
                   note="semacheck.cpp:1081-1090 is a hard Error1 producing NO "
                        "ledger row -- not applicable (specs §2.1)"),
    "M2.13": _spec("M2", "P1", "shape-equal / layout-unequal (every extent "
                               "agrees, the affine map does not)",
                   note="view/metadata family"),
    "M2.14": _spec("M2", "P1", "matmul contraction-dim (K) mismatch masked by "
                               "broadcast"),
    "M2.15": _spec("M2", "P1", "pad_low <-> pad_high SWAPPED (length preserved, "
                               "placement differs)", status="implemented",
                   note="semacheck.cpp:1076-1100 is a SUM -- blind to "
                        "placement; the M2 dual of M2.10"),
    "M2.16": _spec("M2", "P1", "span_as preserving ElementCount() with a "
                               "different rank/split", status="implemented",
                   note="semacheck.cpp:947 compares COUNT, not shape"),
    "M2.17": _spec("M2", "P3", "span_as / reshape on runtime-shaped data -- "
                               "check skipped entirely", status="implemented",
                   note="semacheck.cpp:946-950 guarded by !RuntimeShaped() on "
                        "both sides"),
    "M2.18": _spec("M2", "P4", "reshape on a non-contiguous span "
                               "(warning-only path)",
                  admissible=False, prohibition="absent",
                  note="semacheck.cpp:1195+ Warning(rop->LOC(), ...) -- the "
                       "only P4 spec, so this gap leaves the warning-then-"
                       "continue path with ZERO coverage. MISSING SURFACE: a "
                       "strided view. Every reshape case applies span_as to "
                       "a contiguous parameter, and preserving "
                       "ElementCount() rules out slicing; a non-contiguous "
                       "span needs a stride the source suite never writes"),
    "M2.19": _spec("M2", "P3", "DMA extent mismatch where >=1 extent is "
                               "symbolic -- hard error escaped",
                  status="implemented",
                  note="semacheck.cpp:1140-1145 -- emit_error forced false"),
    "M2.21": _spec("M2", "P1", "MSB broadcast extent neither 1 nor equal "
                               "(rank-unequal path checks trailing dims only)",
                  note="semacheck.cpp:466-500"),

    # ---- M3 hardware-constraint (16 specs, specs §3) --------------------
    "M3.1": _spec("M3", "P1", "contraction extent not divisible by the "
                              "tensor-core atom (tail dropped)",
                  status="implemented"),
    "M3.2": _spec("M3", "P1", "descriptor dimension >= 2^24 (family A)",
                  admissible=False, prohibition="absent",
                  note="specs §3.3 requires the bound be violated by a STRIDE, "
                       "not a length (a length would allocate). The source "
                       "suite has no strided-view surface, and a descriptor "
                       "extent can only come from `.span_as`, whose "
                       "ElementCount check (M2.16, defect F6) fires first -- "
                       "so a naive realization measures M2.16, not M3.2. "
                       "MISSING SURFACE: a strided view"),
    "M3.3": _spec("M3", "P1", "TMA box byte-size >= 2^24 after 128-byte "
                              "ceiling (family B)",
                  admissible=False, prohibition="absent",
                  note="family B needs a 128-byte-ceiled row extent >= 2^24, "
                       "i.e. a row of >= 2^22 f32. Reaching it while "
                       "preserving ElementCount() is arithmetically "
                       "impossible (the row would exceed the whole tensor), "
                       "so every realization either allocates or trips the "
                       "count check. MISSING SURFACE: a strided view"),
    "M3.4": _spec("M3", "P1", "tensor footprint >= 4 GB, 5-D product "
                              "(family C)",
                  admissible=False, prohibition="absent",
                  note="family C needs 5 dims, each small, with a product "
                       ">= 2^30 elements. The suite's maximum rank is 4. "
                       "MISSING SURFACE: a rank-5 case"),
    "M3.5": _spec("M3", "P1", "swizzle-incompatible box shape",
                  admissible=False, prohibition="absent",
                  note="MISSING SURFACE: no case in the suite writes an "
                       "explicit swizzle mode, so swizzle width <-> box "
                       "geometry is not source-expressible"),
    "M3.6": _spec("M3", "P1", "leading dim not aligned to descriptor "
                              "granularity ON A VECTORIZED ACCESS",
                  admissible=False, prohibition="absent",
                  status="implemented",
                  note="replaces v1 M3 s2: the v1 realization (lhs1/rhs1/w1/i1) "
                       "is a noop because the reference k_matmul is SCALAR, so "
                       "an unaligned scalar load is legal. Retained as a named "
                       "control until the vectorized realization is written."),
    "M3.7": _spec("M3", "P1", "TMA inner-box geometry not 128-bit aligned",
                  admissible=False, prohibition="absent",
                  note="MISSING SURFACE: box geometry is inferred by the "
                       "lowering, never written in source; the only direct "
                       "lever is an extent change, which allocates"),
    "M3.8": _spec("M3", "P1", "DMA rank = 6 (outside the assessed [1,5])",
                  admissible=False, prohibition="absent",
                  note="MISSING SURFACE: a rank-6 DMA needs a rank-6 tensor; "
                       "the suite's maximum rank is 4"),
    "M3.9": _spec("M3", "P1", "pad-field overrun (dma.pad / padding_mid beyond "
                              "the assessed range)"),
    "M3.10": _spec("M3", "P1", "last-dim / rank-5 mid-padding violates "
                               "padding_mid[rank-1] == 0"),
    "M3.11": _spec("M3", "P1", "shared operand base not 128-byte aligned on "
                               "sm_90+",
                   admissible=False, prohibition="absent",
                   note="ARCH-DEPENDENT: a noop on sm_86 (SHARED alignment "
                        "16), a mis-read on sm_90+ (128); needs per-mutant "
                        "-arch (specs §9.4 q7). MISSING SURFACE: the suite "
                        "never writes an explicit shared base offset, so the "
                        "alignment cannot be perturbed from source"),
    "M3.12": _spec("M3", "P2", "shared tile exactly at the capacity bound "
                               "(+/-1 KiB edge)",
                   admissible=False, prohibition="repaired",
                   note="RECLASSIFIED P1 -> P2. memcheck.hpp:106-123 "
                        "(CheckCtMemUsage) raises Error1 when compile-time "
                        "SHARED/LOCAL usage exceeds the limit, so the model "
                        "FORBIDS the state and the compiler refuses it: there "
                        "is no test to run. A runtime channel exists only when "
                        "an extent is symbolic, which the suite's shared "
                        "tiles are not. This also explains the 28/40 M3 noops "
                        "in the committed v1 baseline -- v1 s3 was exactly "
                        "this family (specs §3.0)"),
    "M3.13": _spec("M3", "P1", "swizzle width <-> box inner dim <-> shared "
                               "alignment, narrowed",
                   admissible=False, prohibition="absent",
                   note="the injected state must SURVIVE the compiler's own "
                        "repair at cute_codegen.cpp:10271 -- only the "
                        "sub-case the repair misses is injectable (R3), and "
                        "no case in the suite writes an explicit swizzle. "
                        "MISSING SURFACE: a swizzled shared descriptor"),
    "M3.14": _spec("M3", "P3", "linear .copy with a dimension >= 2^24 -- the "
                               "check is absent", status="implemented",
                   note="gpu_adapt.hpp:320 `// linear copy` ... `// omitted`; "
                        "the other six cells of the DMA matrix call "
                        "CheckDimSize (defect F1)"),
    "M3.15": _spec("M3", "P3", ".pad with a dimension >= 2^24 -- the pad path "
                               "never calls CheckDimSize", status="implemented",
                   note="gpu_adapt.hpp:360-450 (defect F2)"),
    "M3.16": _spec("M3", "P3", "TMA box inner alignment with a SYMBOLIC leading "
                               "dim -- no assessment", status="implemented",
                   note="gpu_adapt.hpp:640 `// TODO: emit runtime assessment` "
                        "(defect F3)"),

    # ---- M4 iteration-validity (5 specs, specs §9.1) --------------------
    "M4.1": _spec("M4", "P1", "with-in mdspan dim mutated to 0 -- control",
                  note="LoopBound, forced to ENTRY cost, enabled 11/11"),
    "M4.2": _spec("M4", "P1", "parallelby bound mutated to 0 or negative"),
    "M4.3": _spec("M4", "P1", "parallelby bound symbolic and zero only at "
                              "runtime"),
    "M4.4": _spec("M4", "P1", "bound > 0 but the iteration space is empty "
                              "(zero-trip loop)", admissible=False,
                  prohibition="absent",
                  note="deliberate noop control -- there is nothing to "
                       "forbid: a zero-trip loop is legal and its body never "
                       "executes. If it is ever counted as admissible the "
                       "oracle has regressed"),
    "M4.5": _spec("M4", "P1", "stride/step = 0 in an iteration"),

    # ---- L launch-status (10 specs, specs §4) ---------------------------
    # Attribution-only: these never enter an admissible denominator. Their value
    # is that they grow the never-attribution table, showing that a miss is
    # usually a REJECTED LAUNCH rather than a wrong answer.
    "L1": _spec("L", "L", "shared-memory tile exceeds the device limit",
                admissible=False, prohibition="observation",
                status="implemented",
                note="migrated out of v1 M3 s3 (shared32/64/128, "
                     "shared512/1024); observed as launch rejected. KEPT "
                     "VERBATIM: the committed E1 baseline defines this "
                     "realization, and a reclassification would invalidate "
                     "the frozen S1/S2. The memcheck Error1 finding under "
                     "M3.12 argues it should be re-examined in v3"),
    "L2": _spec("L", "L", "thread-block / cluster extent exceeds the device "
                          "limit", admissible=False,
                prohibition="observation", status="implemented",
                note="hosts v1 M3 s3's per-thread accumulator oversize "
                     "(local8/local16): same observation channel (launch "
                     "rejected), the §4 text names block/cluster extent. "
                     "KEPT VERBATIM for the same reason as L1"),
    "L3": _spec("L", "L", "__launch_bounds__ understated vs actual block size",
                admissible=False, prohibition="absent",
                note="MISSING SURFACE: no case in the suite writes "
                     "__launch_bounds__"),
    "L4": _spec("L", "L", "block extent not a multiple of 32 / of 128",
                admissible=False, prohibition="observation",
                note="the one L spec with an operator: the block extent is "
                     "source-visible (`parallel p by N`), so this class is "
                     "exercised end to end"),
    "L5": _spec("L", "P2", "shared tile exceeds per-SM capacity",
                admissible=False, prohibition="repaired",
                note="RECLASSIFIED: the v2.1 L-class membership assumed the "
                     "launch-rejected channel, but the check the compiler "
                     "actually performs is memcheck.hpp:106-123 CheckCtMemUsage "
                     "-> Error1, a hard compile-time refusal (P2). Same "
                     "mechanism as M3.12, one step past the limit"),
    "L6": _spec("L", "L", "WGMMA used below sm_90", admissible=False,
                prohibition="absent",
                note="MISSING SURFACE: no case in the suite emits an explicit "
                     "WGMMA"),
    "L7": _spec("L", "L", "shared operand used outside WGMMA", admissible=False,
                prohibition="absent",
                note="MISSING SURFACE: no WGMMA in the suite, so 'outside "
                     "WGMMA' has no referent"),
    "L8": _spec("L", "L", "unsupported MMA configuration", admissible=False,
                prohibition="absent",
                note="MISSING SURFACE: no explicit MMA in the suite -- the "
                     "reference k_matmul is a scalar helper, not a tensor-"
                     "core intrinsic"),
    "L9": _spec("L", "L", "mma.scale operand is not an accumulator",
                admissible=False, prohibition="absent",
                note="MISSING SURFACE: no mma.scale in the DSL surface the "
                     "suite uses"),
    "L10": _spec("L", "L", "cluster extent > 8 (non-portable)", admissible=False,
                 prohibition="absent",
                 note="MISSING SURFACE: no case in the suite declares a "
                      "cluster"),
}

# The v1 spec integers still carried by every operator (specs v1 §1-§3) map 1:1
# onto v2.1 ids for M1 and M2, and onto a REVISED target for M3:
#   M3 s1 -> M3.1  (retained verbatim)
#   M3 s2 -> M3.6  (noop; the vectorized realization is M3.6)
#   M3 s3 -> L1/L2 (resource-exhaustion, reclassified OUT of M3 -- specs §3.0)
# Prefer an explicit `spec_id=` on new operators; this table exists so the v1
# corpus stays attributable without rewriting 117 call sites.
V1_SPEC_ID = {
    ("M1", 1): "M1.1", ("M1", 2): "M1.2", ("M1", 3): "M1.3",
    ("M1", 4): "M1.4", ("M1", 5): "M1.5", ("M1", 6): "M1.6",
    ("M2", 1): "M2.1", ("M2", 2): "M2.2", ("M2", 3): "M2.3",
    ("M2", 4): "M2.4", ("M2", 5): "M2.5",
    ("M3", 1): "M3.1", ("M3", 2): "M3.6", ("M3", 3): "L1",
}


class Mut:
    """One defect instance: an ordered list of literal textual edits.

    Each edit is (old, new, n) where n is the required occurrence count in the
    target source, or None meaning "at least one, replace all". Requiring an
    exact count is what stops a transform from rewriting sites it was not
    designed for.

    v2.1 metadata (specs §9.6, §9.5.0) travels with the instance:

      spec_id       v2.1 id, e.g. "M1.20"; the field new tables key on
      spec          the v1 integer, retained so v1-era results stay comparable
      path_class    P1 | P2 | P3 | P4 | L
      prohibition   "" unless the spec is not applicable
      admissible    may this mutant enter the admissible denominator?
      level         1 or 2 (widening level, specs §5)
    """

    def __init__(self, mid, cls, spec, paper_category, category, case, desc,
                 edits, spec_id=None, path_class=None, prohibition=None,
                 admissible=None, level=1):
        self.id = mid
        self.cls = cls
        self.spec = spec
        self.paper_category = paper_category
        self.category = category
        self.case = case
        self.desc = desc
        self.edits = edits
        self.level = level

        # ---- v2.1 attribution, resolved from the registry ---------------
        # An explicit `spec_id=` wins; else the v1-integer map (which carries
        # the M3 splits); else the identity "cls.spec", which is what every
        # spec added after v2.1 uses.
        sid = (spec_id or V1_SPEC_ID.get((cls, spec)) or "%s.%s" % (cls, spec))
        # v1 M3 s3 is resource-exhaustion reclassified out of M3 (specs §3.0).
        # The observation channel splits it: shared -> L1, per-thread -> L2.
        if sid == "L1" and "local" in mid:
            sid = "L2"
        self.spec_id = sid
        meta = SPEC_REGISTRY.get(sid) if sid else None
        if meta is None:
            raise ValueError(
                "mutation %r names spec_id %r, which is not in SPEC_REGISTRY; "
                "add it (specs v2.1 §1-§4) before generating" % (mid, sid))
        self.path_class = path_class or meta["path"]
        self.prohibition = (meta["prohibition"] if prohibition is None
                            else prohibition)
        self.spec_admissible = meta["admissible"]
        if admissible is None:
            admissible = meta["admissible"] and sid not in NOOP_BY_CONSTRUCTION
        self.admissible = admissible

    @property
    def is_na(self):
        """Not applicable: the model FORBIDS the corrupted state, so the
        compiler repairs it and there is no test to run (specs §9.5.0).

        Only P2. Admissibility is a separate, reporting-only concept: an L
        operator IS generated (it grows the never-attribution table) and is
        merely excluded from the admissible denominator.
        """
        return self.path_class == "P2"

    @property
    def needs_rtc_curve(self):
        """P1 is the only path a threshold can reach (specs §9.6.1)."""
        return self.path_class == "P1"

    def as_meta(self):
        """The v2.1 fields, for the manifest and the register."""
        return {
            "spec": self.spec,
            "spec_id": self.spec_id,
            "path_class": self.path_class,
            "prohibition": self.prohibition,
            "admissible": bool(self.admissible),
            "spec_admissible": bool(self.spec_admissible),
            "level": self.level,
        }


def _m(mid, cls, spec, pc, cat, case, desc, *edits, **kw):
    """Shorthand: edits are (old, new) or (old, new, n).

    Keyword arguments are the v2.1 metadata (spec_id, path_class, prohibition,
    admissible, level) and are forwarded to `Mut`.
    """
    norm = [e if len(e) == 3 else (e[0], e[1], None) for e in edits]
    return Mut(mid, cls, spec, pc, cat, case, desc, norm, **kw)


# ===========================================================================
# M1 -- element-access (specs §1)
# Paper categories folded in: input-dependent OOB + stride/layout error.
# Specs: 1 dropped boundary mask, 2 off-by-one, 3 negative index,
#        4 transposed/non-contiguous stride, 5 offset view, 6 zero-stride/empty.
# ===========================================================================
M1 = [
    # ---- layer_normalization / 1_bert (static) -------------------------
    _m("M1.s2.ln1.scale", "M1", 2, "oob", "layer_normalization",
       "1_bert_32x512x768_768_768",
       "off-by-one on the secondary operand index: scale.at(k+1) is OOB at k == K-1",
       ("scale.at(k)", "scale.at(k + 1)")),
    _m("M1.s3.ln1.bias", "M1", 3, "oob", "layer_normalization",
       "1_bert_32x512x768_768_768",
       "negative index: bias.at(k-1) underflows at k == 0",
       ("bias.at(k)", "bias.at(k - 1)")),
    _m("M1.s5.ln1.chunk", "M1", 5, "oob", "layer_normalization",
       "1_bert_32x512x768_768_768",
       "offset view: advance the chunked parallel index past the last block",
       ("lhs.at(p#n, j, k)", "lhs.at(p#n + 1, j, k)")),
    _m("M1.s6.ln1.empty", "M1", 6, "stride", "layer_normalization",
       "1_bert_32x512x768_768_768",
       "empty iteration range (iteration-validity residue)",
       ("foreach {j, k} in [J, K]", "foreach {j, k} in [J, 0]")),
    _m("M1.s6.ln1.zerostride", "M1", 6, "stride", "layer_normalization",
       "1_bert_32x512x768_768_768",
       "zero-stride access: collapse the varying k index to a constant",
       ("lhs.at(p#n, j, k)", "lhs.at(p#n, j, 0)")),
    _m("M1.s2.ln1.out", "M1", 2, "oob", "layer_normalization",
       "1_bert_32x512x768_768_768",
       "off-by-one on the output write index",
       ("out.at(p#n, j, k)", "out.at(p#n, j, k + 1)")),

    # ---- layer_normalization / 3_attention (dynamic) -------------------
    _m("M1.s2.ln3.scale", "M1", 2, "oob", "layer_normalization",
       "3_attention_32xNx512x64_64_64",
       "off-by-one on the secondary operand index (dynamic extent L)",
       ("scale.at(l)", "scale.at(l + 1)")),
    _m("M1.s3.ln3.bias", "M1", 3, "oob", "layer_normalization",
       "3_attention_32xNx512x64_64_64",
       "negative index on the secondary operand",
       ("bias.at(l)", "bias.at(l - 1)")),
    _m("M1.s1.ln3.mask", "M1", 1, "oob", "layer_normalization",
       "3_attention_32xNx512x64_64_64",
       "dropped boundary guard: advance a dynamic loop index past its extent",
       ("lhs.at(p#n, j, k, l)", "lhs.at(p#n, j + 1, k, l)")),
    _m("M1.s6.ln3.empty", "M1", 6, "stride", "layer_normalization",
       "3_attention_32xNx512x64_64_64",
       "empty reduction range over the dynamic extent",
       ("foreach l in [L]", "foreach l in [0]")),
    _m("M1.s5.ln3.out", "M1", 5, "oob", "layer_normalization",
       "3_attention_32xNx512x64_64_64",
       "offset view on the output chunk index",
       ("out.at(p#n, j, k, l)", "out.at(p#n + 1, j, k, l)")),
    _m("M1.s6.ln3.overrange", "M1", 6, "stride", "layer_normalization",
       "3_attention_32xNx512x64_64_64",
       "iteration range one past the symbolic extent",
       ("foreach {j, k} in [N0, K]", "foreach {j, k} in [N0, K + 1]")),

    # ---- softmax / 1_bert (static) -------------------------------------
    _m("M1.s2.sm1.read", "M1", 2, "oob", "softmax",
       "1_bert_32x512x768_32x512x768",
       "off-by-one on the reduction read index",
       ("l1_input.data.at(0, j, k)", "l1_input.data.at(0, j, k + 1)")),
    _m("M1.s3.sm1.write", "M1", 3, "oob", "softmax",
       "1_bert_32x512x768_32x512x768",
       "negative row index into the shared output tile",
       ("l1_out.at(0, j, k)", "l1_out.at(0, j - 1, k)")),
    _m("M1.s6.sm1.empty", "M1", 6, "stride", "softmax",
       "1_bert_32x512x768_32x512x768",
       "empty reduction range",
       ("foreach {k} in [l1_input.span(2)]", "foreach {k} in [0]")),
    _m("M1.s5.sm1.store", "M1", 5, "oob", "softmax",
       "1_bert_32x512x768_32x512x768",
       "offset write-back chunk overruns the output tensor",
       ("dma.copy l1_out => output.chunkat(i#p, q, _)",
        "dma.copy l1_out => output.chunkat(i#p + 1, q, _)")),
    _m("M1.s6.sm1.zerostride", "M1", 6, "stride", "softmax",
       "1_bert_32x512x768_32x512x768",
       "zero-stride access: collapse the reduction index",
       ("l1_input.data.at(0, j, k)", "l1_input.data.at(0, j, 0)")),

    # ---- softmax / 11_dynamic ------------------------------------------
    _m("M1.s2.sm11.read", "M1", 2, "oob", "softmax",
       "11_dynamic_32xSx768_32xSx768",
       "off-by-one on the reduction read index (dynamic seq_len)",
       ("l1_input.data.at(0, j, k)", "l1_input.data.at(0, j, k + 1)")),
    _m("M1.s5.sm11.write", "M1", 5, "oob", "softmax",
       "11_dynamic_32xSx768_32xSx768",
       "offset view on the output row index",
       ("l1_out.at(0, j, k)", "l1_out.at(0, j + 1, k)")),
    _m("M1.s6.sm11.empty", "M1", 6, "stride", "softmax",
       "11_dynamic_32xSx768_32xSx768",
       "empty reduction range",
       ("foreach {k} in [l1_input.span(2)]", "foreach {k} in [0]")),
    _m("M1.s5.sm11.store", "M1", 5, "oob", "softmax",
       "11_dynamic_32xSx768_32xSx768",
       "offset write-back chunk overruns the output tensor",
       ("dma.copy l1_out => output.chunkat(i#p, _, _)",
        "dma.copy l1_out => output.chunkat(i#p + 1, _, _)")),
    _m("M1.s1.sm11.mask", "M1", 1, "oob", "softmax",
       "11_dynamic_32xSx768_32xSx768",
       "dropped boundary guard on the row loop over the dynamic extent",
       ("foreach {j} in [l1_input.span(1)]", "foreach {j} in [l1_input.span(1) + 1]")),

    # ---- relu / 1_bert (static) ----------------------------------------
    _m("M1.s2.rl1.read", "M1", 2, "oob", "relu",
       "1_bert_32x512x768_32x512x768",
       "off-by-one on the thread-parallel shared index (inp_s is [1,1,64,1])",
       ("inp_s.at(0, 0, q, 0) > 0.0f", "inp_s.at(0, 0, q + 1, 0) > 0.0f")),
    _m("M1.s3.rl1.write", "M1", 3, "oob", "relu",
       "1_bert_32x512x768_32x512x768",
       "negative index on the shared output tile",
       ("out_s.at(0, 0, q, 0) =", "out_s.at(0, 0, q - 1, 0) =")),
    _m("M1.s6.rl1.empty", "M1", 6, "stride", "relu",
       "1_bert_32x512x768_32x512x768",
       "empty tile range on the innermost foreach",
       ("foreach {i, j, k} in [I / #p, J, 12]", "foreach {i, j, k} in [I / #p, J, 0]")),
    _m("M1.s5.rl1.load", "M1", 5, "oob", "relu",
       "1_bert_32x512x768_32x512x768",
       "offset async DMA source chunk overruns the input",
       ("inp.chunkat(p#i, j, k, _)", "inp.chunkat(p#i + 1, j, k, _)")),
    _m("M1.s5.rl1.store", "M1", 5, "oob", "relu",
       "1_bert_32x512x768_32x512x768",
       "offset DMA destination chunk overruns the output",
       ("dma.copy out_s => out.chunkat(p#i, j, k, _)",
        "dma.copy out_s => out.chunkat(p#i, j, k + 1, _)")),
    _m("M1.s4.rl1.tile", "M1", 4, "stride", "relu",
       "1_bert_32x512x768_32x512x768",
       "non-contiguous shared tile: extent 64 -> 65 against `parallel q by 64`",
       ("shared f32 [1, 1, 64, 1] inp_s, out_s;", "shared f32 [1, 1, 65, 1] inp_s, out_s;")),

    # ---- relu / 11_dynamic ---------------------------------------------
    _m("M1.s2.rl11.read", "M1", 2, "oob", "relu",
       "11_dynamic_32xSx768_32xSx768",
       "off-by-one on the thread-parallel shared index",
       ("inp_s.at(0, 0, q, 0) > 0.0f", "inp_s.at(0, 0, q + 1, 0) > 0.0f")),
    _m("M1.s3.rl11.write", "M1", 3, "oob", "relu",
       "11_dynamic_32xSx768_32xSx768",
       "negative index on the shared output tile",
       ("out_s.at(0, 0, q, 0) =", "out_s.at(0, 0, q - 1, 0) =")),
    _m("M1.s6.rl11.empty", "M1", 6, "stride", "relu",
       "11_dynamic_32xSx768_32xSx768",
       "empty tile range on the innermost foreach",
       ("foreach {i, j, k} in [I / #p, NUM_HEADS, 12]",
        "foreach {i, j, k} in [I / #p, NUM_HEADS, 0]")),
    _m("M1.s5.rl11.load", "M1", 5, "oob", "relu",
       "11_dynamic_32xSx768_32xSx768",
       "offset async DMA source chunk overruns the input",
       ("inp.chunkat(p#i, j, k, _)", "inp.chunkat(p#i, j + 1, k, _)")),
    _m("M1.s5.rl11.store", "M1", 5, "oob", "relu",
       "11_dynamic_32xSx768_32xSx768",
       "offset DMA destination chunk overruns the output",
       ("dma.copy out_s => out.chunkat(p#i, j, k, _)",
        "dma.copy out_s => out.chunkat(p#i + 1, j, k, _)")),

    # ---- transpose / 1_bert (static) -----------------------------------
    _m("M1.s2.tp1.read", "M1", 2, "oob", "transpose",
       "1_bert_32x512x768_32x768x512",
       "off-by-one on the shared source tile (is is [1,1,64])",
       ("is.at(0, 0, q)", "is.at(0, 0, q + 1)")),
    _m("M1.s3.tp1.write", "M1", 3, "oob", "transpose",
       "1_bert_32x512x768_32x768x512",
       "negative index on the shared destination tile",
       ("os.at(0, q, 0)", "os.at(0, q - 1, 0)")),
    _m("M1.s6.tp1.empty", "M1", 6, "stride", "transpose",
       "1_bert_32x512x768_32x768x512",
       "empty inner tile range",
       ("foreach {y, z} in [512, 12]", "foreach {y, z} in [512, 0]")),
    _m("M1.s5.tp1.load", "M1", 5, "oob", "transpose",
       "1_bert_32x512x768_32x768x512",
       "offset async DMA source chunk overruns the input",
       ("i.chunkat(p, y, z)", "i.chunkat(p + 1, y, z)")),
    _m("M1.s5.tp1.store", "M1", 5, "oob", "transpose",
       "1_bert_32x512x768_32x768x512",
       "offset DMA destination chunk overruns the output",
       ("dma.copy os => o.chunkat(p, z, y)", "dma.copy os => o.chunkat(p, z, y + 1)")),
    _m("M1.s4.tp1.layout", "M1", 4, "stride", "transpose",
       "1_bert_32x512x768_32x768x512",
       "transposed output layout: undo the permutation in the declaration",
       ("f32 [i.span(0), i.span(2), i.span(1)] o;",
        "f32 [i.span(0), i.span(1), i.span(2)] o;")),
    _m("M1.s4.tp1.tile", "M1", 4, "stride", "transpose",
       "1_bert_32x512x768_32x768x512",
       "non-contiguous shared tile: extent 64 -> 65 against `parallel q by 64`",
       ("shared f32 [1, 1, 64] is;", "shared f32 [1, 1, 65] is;")),

    # ---- transpose / 11_dynamic ----------------------------------------
    _m("M1.s2.tp11.read", "M1", 2, "oob", "transpose",
       "11_dynamic_32xSx768_32x768xS",
       "off-by-one on the shared source tile",
       ("is.at(0, 0, q)", "is.at(0, 0, q + 1)")),
    _m("M1.s3.tp11.write", "M1", 3, "oob", "transpose",
       "11_dynamic_32xSx768_32x768xS",
       "negative index on the shared destination tile",
       ("os.at(0, q, 0)", "os.at(0, q - 1, 0)")),
    _m("M1.s6.tp11.empty", "M1", 6, "stride", "transpose",
       "11_dynamic_32xSx768_32x768xS",
       "empty inner tile range over the dynamic extent",
       ("foreach {y, z} in [seq_len, 12]", "foreach {y, z} in [seq_len, 0]")),
    _m("M1.s5.tp11.load", "M1", 5, "oob", "transpose",
       "11_dynamic_32xSx768_32x768xS",
       "offset async DMA source chunk overruns the input",
       ("i.chunkat(p, y, z)", "i.chunkat(p, y + 1, z)")),
    _m("M1.s5.tp11.store", "M1", 5, "oob", "transpose",
       "11_dynamic_32xSx768_32x768xS",
       "offset DMA destination chunk overruns the output",
       ("dma.copy os => o.chunkat(p, z, y)", "dma.copy os => o.chunkat(p, z + 1, y)")),
    _m("M1.s4.tp11.layout", "M1", 4, "stride", "transpose",
       "11_dynamic_32xSx768_32x768xS",
       "transposed output layout: undo the permutation in the declaration",
       ("f32 [i.span(0), i.span(2), i.span(1)] o;",
        "f32 [i.span(0), i.span(1), i.span(2)] o;")),
]

# ===========================================================================
# M2 -- shape-compatibility (specs §2)
# Paper categories folded in: dimension-mismatch + wrong-output-shape.
# Specs: 1 wrong leading extent on a secondary operand, 2 DMA source/dest
#        extent disagreement, 3 binary op on mismatched shapes, 4 output
#        declared with a wrong extent, 5 partial / duplicate write.
# ===========================================================================
M2 = [
    # ---- layer_normalization / 1_bert (static) -------------------------
    _m("M2.s1.ln1.scale", "M2", 1, "dim-mismatch", "layer_normalization",
       "1_bert_32x512x768_768_768",
       "secondary operand declared with the wrong leading extent (scale [K] -> [K-1])",
       ("f32 [K] scale", "f32 [K - 1] scale")),
    _m("M2.s1.ln1.bias", "M2", 1, "dim-mismatch", "layer_normalization",
       "1_bert_32x512x768_768_768",
       "secondary operand declared with the wrong leading extent (bias [K] -> [K+1])",
       ("f32 [K] bias", "f32 [K + 1] bias")),
    _m("M2.s4.ln1.out", "M2", 4, "wrong-shape", "layer_normalization",
       "1_bert_32x512x768_768_768",
       "output tensor declared with a wrong trailing extent",
       ("f32 [lhs.span] out;", "f32 [I, J, K - 1] out;")),
    _m("M2.s3.ln1.reduce", "M2", 3, "dim-mismatch", "layer_normalization",
       "1_bert_32x512x768_768_768",
       "reduction divisor disagrees with the reduced extent",
       ("s_mean.at(0) / (lhs.span(1) * lhs.span(2))",
        "s_mean.at(0) / (lhs.span(1) * lhs.span(2) - 1)")),
    _m("M2.s1.ln1.caller.scale", "M2", 1, "dim-mismatch", "layer_normalization",
       "1_bert_32x512x768_768_768",
       "caller materialises the secondary operand with the wrong extent",
       ("auto scale = choreo::make_spandata<choreo::f32>(K);",
        "auto scale = choreo::make_spandata<choreo::f32>(K - 1);")),
    _m("M2.s1.ln1.caller.bias", "M2", 1, "dim-mismatch", "layer_normalization",
       "1_bert_32x512x768_768_768",
       "caller materialises the secondary operand with the wrong extent",
       ("auto bias = choreo::make_spandata<choreo::f32>(K);",
        "auto bias = choreo::make_spandata<choreo::f32>(K + 1);")),
    _m("M2.s4.ln1.caller.out", "M2", 4, "wrong-shape", "layer_normalization",
       "1_bert_32x512x768_768_768",
       "reference output buffer declared with a wrong extent",
       ("auto out = choreo::make_spandata<choreo::f32>(I, J, K);",
        "auto out = choreo::make_spandata<choreo::f32>(I, J, K - 1);")),

    # ---- layer_normalization / 3_attention (dynamic) -------------------
    _m("M2.s1.ln3.scale", "M2", 1, "dim-mismatch", "layer_normalization",
       "3_attention_32xNx512x64_64_64",
       "secondary operand declared with the wrong extent (scale [L] -> [L-1])",
       ("f32 [L] scale", "f32 [L - 1] scale")),
    _m("M2.s1.ln3.bias", "M2", 1, "dim-mismatch", "layer_normalization",
       "3_attention_32xNx512x64_64_64",
       "secondary operand declared with the wrong extent (bias [L] -> [L+1])",
       ("f32 [L] bias", "f32 [L + 1] bias")),
    _m("M2.s4.ln3.out", "M2", 4, "wrong-shape", "layer_normalization",
       "3_attention_32xNx512x64_64_64",
       "output tensor declared with a wrong trailing extent",
       ("f32 [lhs.span] out;", "f32 [I, N0, K, L - 1] out;")),
    _m("M2.s3.ln3.reduce", "M2", 3, "dim-mismatch", "layer_normalization",
       "3_attention_32xNx512x64_64_64",
       "reduction divisor disagrees with the reduced extent",
       ("s_mean.at(0) / lhs.span(3)", "s_mean.at(0) / (lhs.span(3) - 1)")),
    _m("M2.s1.ln3.lhs", "M2", 1, "dim-mismatch", "layer_normalization",
       "3_attention_32xNx512x64_64_64",
       "primary operand declared with a wrong trailing extent",
       ("f32 [I, N0, K, L] lhs", "f32 [I, N0, K, L - 1] lhs")),
    _m("M2.s1.ln3.caller.scale", "M2", 1, "dim-mismatch", "layer_normalization",
       "3_attention_32xNx512x64_64_64",
       "caller materialises the secondary operand with the wrong extent",
       ("auto scale = choreo::make_spandata<choreo::f32>(L);",
        "auto scale = choreo::make_spandata<choreo::f32>(L - 1);")),
    _m("M2.s4.ln3.caller.out", "M2", 4, "wrong-shape", "layer_normalization",
       "3_attention_32xNx512x64_64_64",
       "reference output buffer declared with a wrong extent",
       ("auto out = choreo::make_spandata<choreo::f32>(I, J, K, L);",
        "auto out = choreo::make_spandata<choreo::f32>(I, J, K, L - 1);")),

    # ---- matmul / 1_bert (static) --------------------------------------
    _m("M2.s1.mm1.rhs", "M2", 1, "dim-mismatch", "matmul",
       "1_bert_32x512x768_768x768_32x512x768",
       "rhs operand declared with the wrong trailing extent",
       ("f32 [768, 768] rhs", "f32 [768, 767] rhs")),
    _m("M2.s4.mm1.out", "M2", 4, "wrong-shape", "matmul",
       "1_bert_32x512x768_768x768_32x512x768",
       "output declared with a wrong trailing extent",
       ("f32 [lhs.span(0), lhs.span(1), rhs.span(1)] output;",
        "f32 [lhs.span(0), lhs.span(1), rhs.span(1) - 1] output;")),
    _m("M2.s2.mm1.store", "M2", 2, "dim-mismatch", "matmul",
       "1_bert_32x512x768_768x768_32x512x768",
       "DMA source/destination extent disagreement: transpose the store",
       ("dma.copy l1_out => output.chunkat(p#q, m_tile, n_tile);",
        "dma.transp<1,0> l1_out => output.chunkat(p#q, m_tile, n_tile);")),
    _m("M2.s3.mm1.swap", "M2", 3, "dim-mismatch", "matmul",
       "1_bert_32x512x768_768x768_32x512x768",
       "operand swap: DMA the wrong tensor into the lhs tile",
       ("l1_a = dma.copy lhs.chunkat(p#q, m_tile, k_tile) => local;",
        "l1_a = dma.copy rhs.chunkat(p#q, m_tile, k_tile) => local;")),
    _m("M2.s2.mm1.tiles", "M2", 2, "dim-mismatch", "matmul",
       "1_bert_32x512x768_768x768_32x512x768",
       "tile count disagrees with the tiled extent on the k dimension",
       ("with index = {m_tile, n_tile, k_tile} in [4, 4, 4] {",
        "with index = {m_tile, n_tile, k_tile} in [4, 4, 5] {")),
    _m("M2.s4.mm1.local", "M2", 4, "wrong-shape", "matmul",
       "1_bert_32x512x768_768x768_32x512x768",
       "local accumulator declared with a wrong trailing extent",
       ("output.span(2)/#n_tile] l1_out {0.0f};",
        "output.span(2)/#n_tile - 1] l1_out {0.0f};")),

    # ---- matmul / 11_dynamic -------------------------------------------
    _m("M2.s1.mm11.rhs", "M2", 1, "dim-mismatch", "matmul",
       "11_dynamic_32xSx768_768x768_32xSx768",
       "rhs operand declared with the wrong leading extent",
       ("f32 [768, 768] rhs", "f32 [767, 768] rhs")),
    _m("M2.s4.mm11.out", "M2", 4, "wrong-shape", "matmul",
       "11_dynamic_32xSx768_768x768_32xSx768",
       "output declared with a wrong trailing extent",
       ("f32 [lhs.span(0), lhs.span(1), rhs.span(1)] output;",
        "f32 [lhs.span(0), lhs.span(1), rhs.span(1) - 1] output;")),
    _m("M2.s2.mm11.store", "M2", 2, "dim-mismatch", "matmul",
       "11_dynamic_32xSx768_768x768_32xSx768",
       "DMA source/destination extent disagreement: transpose the store",
       ("dma.copy l1_out => output.chunkat(p#q, m_tile, n_tile);",
        "dma.transp<1,0> l1_out => output.chunkat(p#q, m_tile, n_tile);")),
    _m("M2.s3.mm11.swap", "M2", 3, "dim-mismatch", "matmul",
       "11_dynamic_32xSx768_768x768_32xSx768",
       "operand swap: DMA the wrong tensor into the lhs tile",
       ("l1_a = dma.copy lhs.chunkat(p#q, m_tile, k_tile) => local;",
        "l1_a = dma.copy rhs.chunkat(p#q, m_tile, k_tile) => local;")),
    _m("M2.s2.mm11.tiles", "M2", 2, "dim-mismatch", "matmul",
       "11_dynamic_32xSx768_768x768_32xSx768",
       "tile count disagrees with the tiled extent on the n dimension",
       ("with index = {m_tile, n_tile, k_tile} in [1, 6, 6] {",
        "with index = {m_tile, n_tile, k_tile} in [1, 6, 7] {")),
    _m("M2.s4.mm11.local", "M2", 4, "wrong-shape", "matmul",
       "11_dynamic_32xSx768_768x768_32xSx768",
       "local accumulator declared with a wrong middle extent",
       ("output.span(1)/#m_tile,", "output.span(1)/#m_tile - 1,")),

    # ---- concat / 1_bert (static) --------------------------------------
    _m("M2.s1.cc1.b", "M2", 1, "dim-mismatch", "concat",
       "1_bert_32x512x768_32x512x768_32x512x1536",
       "second operand declared with the wrong extent",
       ("f32 [I, J, K, L] b)", "f32 [I, J, K - 1, L] b)")),
    _m("M2.s1.cc1.a", "M2", 1, "dim-mismatch", "concat",
       "1_bert_32x512x768_32x512x768_32x512x1536",
       "first operand declared with the wrong extent",
       ("f32 [I, J, K, L] a,", "f32 [I, J, K + 1, L] a,")),
    _m("M2.s4.cc1.out", "M2", 4, "wrong-shape", "concat",
       "1_bert_32x512x768_32x512x768_32x512x1536",
       "output declared with a wrong concatenation extent (signature and body)",
       ("f32 [I, J, 1536, L]", "f32 [I, J, 1535, L]")),
    _m("M2.s2.cc1.tile", "M2", 2, "dim-mismatch", "concat",
       "1_bert_32x512x768_32x512x768_32x512x1536",
       "shared destination tile smaller than the DMA source extent",
       ("shared f32 [1, 1, 2 * K, L] os;", "shared f32 [1, 1, 2 * K - 1, L] os;")),
    _m("M2.s5.cc1.dup", "M2", 5, "wrong-shape", "concat",
       "1_bert_32x512x768_32x512x768_32x512x1536",
       "duplicate write: second half overwrites the first (overlapping tile)",
       ("os.at(0, 0, q + K, 0) = bs.at(0, 0, q, 0);",
        "os.at(0, 0, q, 0) = bs.at(0, 0, q, 0);")),
    _m("M2.s5.cc1.partial", "M2", 5, "wrong-shape", "concat",
       "1_bert_32x512x768_32x512x768_32x512x1536",
       "partial write: omitted tail element of the concatenation",
       ("foreach q in [K] {", "foreach q in [K - 1] {")),
    _m("M2.s2.cc1.as", "M2", 2, "dim-mismatch", "concat",
       "1_bert_32x512x768_32x512x768_32x512x1536",
       "shared source tile extent disagrees with the DMA source",
       ("shared f32 [1, 1, K, L] as, bs;", "shared f32 [1, 1, K + 1, L] as, bs;")),

    # ---- concat / 11_dynamic -------------------------------------------
    _m("M2.s1.cc11.b", "M2", 1, "dim-mismatch", "concat",
       "11_dynamic_32xS1x768_32xS2x768_32xS1pS2x768",
       "second operand declared with the wrong extent",
       ("f32 [I, J2, K, L] b)", "f32 [I, J2 - 1, K, L] b)")),
    _m("M2.s1.cc11.a", "M2", 1, "dim-mismatch", "concat",
       "11_dynamic_32xS1x768_32xS2x768_32xS1pS2x768",
       "first operand declared with the wrong extent",
       ("f32 [I, J1, K, L] a,", "f32 [I, J1 + 1, K, L] a,")),
    _m("M2.s4.cc11.out", "M2", 4, "wrong-shape", "concat",
       "11_dynamic_32xS1x768_32xS2x768_32xS1pS2x768",
       "output declared with a wrong concatenation extent",
       ("f32 [I, J_OUT, K, L] out;", "f32 [I, J_OUT - 1, K, L] out;")),
    _m("M2.s2.cc11.tile", "M2", 2, "dim-mismatch", "concat",
       "11_dynamic_32xS1x768_32xS2x768_32xS1pS2x768",
       "shared destination tile smaller than the concatenated extent",
       ("shared f32 [1, J_OUT, 1, 1] os;", "shared f32 [1, J_OUT - 1, 1, 1] os;")),
    _m("M2.s5.cc11.dup", "M2", 5, "wrong-shape", "concat",
       "11_dynamic_32xS1x768_32xS2x768_32xS1pS2x768",
       "duplicate write: second half overwrites the first",
       ("os.at(0, J1 + q, 0, 0) = bs.at(0, q, 0, 0);",
        "os.at(0, q, 0, 0) = bs.at(0, q, 0, 0);")),
    _m("M2.s5.cc11.partial", "M2", 5, "wrong-shape", "concat",
       "11_dynamic_32xS1x768_32xS2x768_32xS1pS2x768",
       "partial write: omitted tail element of the concatenation",
       ("foreach q in [J2]", "foreach q in [J2 - 1]")),
    _m("M2.s2.cc11.as", "M2", 2, "dim-mismatch", "concat",
       "11_dynamic_32xS1x768_32xS2x768_32xS1pS2x768",
       "shared source tile extent disagrees with the DMA source",
       ("shared f32 [1, J1, 1, 1] as;", "shared f32 [1, J1 + 1, 1, 1] as;")),
]

# ===========================================================================
# M3 -- hardware-constraint (specs §3)
# No RQ2 paper category; in scope by the ledger's hw-constraint usage class and
# by rathnasuriya2026tilebugs (WGMMA shape-divisibility + resource alignment).
# Specs: 1 contraction dim not divisible by the tensor-core atom, 2 misaligned
#        tensor-core / DMA base, 3 shared-memory tile over the device limit.
# Counts are reached by enumerating residues / offsets / sizes within a spec.
# ===========================================================================
M3 = []

# ---- matmul / 1_bert (static): contraction arg to the MMA atom -----------
_MM1_CALL = ("call k_matmul(l1_a.chunkat(i,_,_), l1_b.data, l1_out, "
             "l1_a.span(1), l1_b.span(0), l1_a.span(2));")
for _k in (6, 10, 12, 20, 24):          # all % 16 != 0
    M3.append(_m(f"M3.s1.mm1.atom{_k}", "M3", 1, "dim-mismatch", "matmul",
                 "1_bert_32x512x768_768x768_32x512x768",
                 f"MMA contraction extent {_k} not divisible by the tensor-core atom (16)",
                 (_MM1_CALL, _MM1_CALL.replace("l1_a.span(2));", f"{_k});"))))
for _t in (7, 9):                        # 768 / t is not an integer multiple of 16
    M3.append(_m(f"M3.s1.mm1.ktile{_t}", "M3", 1, "dim-mismatch", "matmul",
                 "1_bert_32x512x768_768x768_32x512x768",
                 f"k-tile count {_t} yields a contraction tile that is not atom-divisible",
                 ("with index = {m_tile, n_tile, k_tile} in [4, 4, 4] {",
                  "with index = {m_tile, n_tile, k_tile} in [4, 4, %d] {" % _t)))
for _off in (1, 2):
    M3.append(_m(f"M3.s2.mm1.rhs{_off}", "M3", 2, "stride", "matmul",
                 "1_bert_32x512x768_768x768_32x512x768",
                 f"misaligned tensor-core base: rhs DMA chunk offset by {_off}",
                 ("l1_b = dma.transp<1,0> rhs.chunkat(k_tile, n_tile) => local;",
                  f"l1_b = dma.transp<1,0> rhs.chunkat(k_tile, n_tile + {_off}) => local;")))
M3.append(_m("M3.s2.mm1.lhs1", "M3", 2, "stride", "matmul",
             "1_bert_32x512x768_768x768_32x512x768",
             "misaligned DMA base: lhs chunk offset on the contraction tile",
             ("l1_a = dma.copy lhs.chunkat(p#q, m_tile, k_tile) => local;",
              "l1_a = dma.copy lhs.chunkat(p#q, m_tile, k_tile + 1) => local;")))
for _mul in (8, 16):
    M3.append(_m(f"M3.s3.mm1.local{_mul}", "M3", 3, "dim-mismatch", "matmul",
                 "1_bert_32x512x768_768x768_32x512x768",
                 f"per-thread accumulator tile {_mul}x over the local-memory budget",
                 ("output.span(2)/#n_tile] l1_out {0.0f};",
                  "output.span(2)/#n_tile * %d] l1_out {0.0f};" % _mul)))

# ---- matmul / 11_dynamic -------------------------------------------------
_MM11_CALL = ("call k_matmul(l1_a.chunkat(i,_,_), l1_b.data, l1_out, "
              "l1_a.span(1), l1_b.span(0), l1_a.span(2));")
for _k in (6, 12, 20):
    M3.append(_m(f"M3.s1.mm11.atom{_k}", "M3", 1, "dim-mismatch", "matmul",
                 "11_dynamic_32xSx768_768x768_32xSx768",
                 f"MMA contraction extent {_k} not divisible by the tensor-core atom (16)",
                 (_MM11_CALL, _MM11_CALL.replace("l1_a.span(2));", f"{_k});"))))
M3.append(_m("M3.s1.mm11.ktile31", "M3", 1, "dim-mismatch", "matmul",
             "11_dynamic_32xSx768_768x768_32xSx768",
             "k-tile count 31 over a SYMBOLIC contraction extent: the per-tile "
             "K = span(2)/31 is not statically atom-divisible, so the "
             "divisibility obligation never reaches runtime -- the dynamic "
             "twin of M3.s1.mm1.atom*6, which is caught statically",
             ("with index = {m_tile, n_tile, k_tile} in [1, 32, 32] {",
              "with index = {m_tile, n_tile, k_tile} in [1, 32, 31] {")))
M3.append(_m("M3.s2.mm11.rhs1", "M3", 2, "stride", "matmul",
             "11_dynamic_32xSx768_768x768_32xSx768",
             "misaligned tensor-core base: rhs DMA chunk offset by 1",
             ("l1_b = dma.transp<1,0> rhs.chunkat(k_tile, n_tile) => local;",
              "l1_b = dma.transp<1,0> rhs.chunkat(k_tile, n_tile + 1) => local;")))
M3.append(_m("M3.s3.mm11.local8", "M3", 3, "dim-mismatch", "matmul",
             "11_dynamic_32xSx768_768x768_32xSx768",
             "per-thread accumulator tile 8x over the per-thread budget",
             ("output.span(2)/#n_tile] l1_out {0.0f};",
              "output.span(2)/#n_tile * 8] l1_out {0.0f};")))

# ---- conv2d / 1_attention (Cout=1024, K=Cin*Kh*Kw=512, k-tile 8) ---------
_C1 = "1_attention_32x512xHxW_1024x512x1x1_32x1024xHxW_1_0_1"
for _d in (2, 4, 6, 8, 12):              # 512 + d is never % 16 == 0
    M3.append(_m(f"M3.s1.cv1.kplus{_d}", "M3", 1, "dim-mismatch", "conv2d", _C1,
                 f"contraction extent K + {_d} not divisible by the tensor-core atom (16)",
                 ("K = Cin * Kh * Kw;", f"K = Cin * Kh * Kw + {_d};")))
_C1_CALL = "call k_matmul(l1_A.data, l1_B.data, l1_Y, M/#q, Cout, 8);"
for _k in (6, 10, 12, 20):
    M3.append(_m(f"M3.s1.cv1.atom{_k}", "M3", 1, "dim-mismatch", "conv2d", _C1,
                 f"MMA contraction extent {_k} not divisible by the tensor-core atom (16)",
                 (_C1_CALL, _C1_CALL.replace("Cout, 8);", f"Cout, {_k});"))))
for _w in (32, 64, 128):
    # 1024 * w * 4 B = 128 KB / 256 KB / 512 KB, all over the 48 KB shared limit.
    # The k-tile loop and the atom extent move with it so the defect stays purely
    # "shared tile too large" rather than becoming an M2 extent disagreement.
    M3.append(_m(f"M3.s3.cv1.shared{_w}", "M3", 3, "dim-mismatch", "conv2d", _C1,
                 f"shared-memory weight tile 1024 x {_w} f32 = {1024 * _w * 4 // 1024} KB, "
                 f"over the 48 KB device limit",
                 ("shared f32 [Cout, 8] B_tile;", f"shared f32 [Cout, {_w}] B_tile;"),
                 ("foreach kt in [K / 8] {", "foreach kt in [K / %d] {" % _w),
                 (_C1_CALL, _C1_CALL.replace("Cout, 8);", f"Cout, {_w});"))))
M3.append(_m("M3.s2.cv1.w1", "M3", 2, "stride", "conv2d", _C1,
             "misaligned DMA base address on the weight tensor",
             ("dma.copy w.span_as(Cout, K).chunkat(_, kt) => B_tile;",
              "dma.copy w.span_as(Cout, K).chunkat(_, kt + 1) => B_tile;")))
M3.append(_m("M3.s2.cv1.i1", "M3", 2, "stride", "conv2d", _C1,
             "misaligned DMA base address on the input tensor",
             ("i.chunkat(p#n, _, _, _).span_as(K, Ho, Wo)",
              "i.chunkat(p#n + 1, _, _, _).span_as(K, Ho, Wo)")))
M3.append(_m("M3.s3.cv1.local8", "M3", 3, "dim-mismatch", "conv2d", _C1,
             "local accumulator tile 8x over the per-thread budget",
             ("local f32 [M/#q, Cout] l1_Y{0.0f};",
              "local f32 [M/#q, Cout * 8] l1_Y{0.0f};")))

# ---- conv2d / 11_static (Cout=32, Cin=128, K=128, full weight in shared) --
_C2 = "11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1"
for _d in (4, 8, 12):                    # 128 + d is never % 16 == 0
    M3.append(_m(f"M3.s1.cv2.kplus{_d}", "M3", 1, "dim-mismatch", "conv2d", _C2,
                 f"contraction extent K + {_d} not divisible by the tensor-core atom (16)",
                 ("K = Cin * Kh * Kw;", f"K = Cin * Kh * Kw + {_d};")))
_C2_CALL = "call k_matmul(l1_A.data, l1_B.data, l1_Y, M/#q, Cout/#qq, K);"
for _d in (4, 12):
    M3.append(_m(f"M3.s1.cv2.atom{_d}", "M3", 1, "dim-mismatch", "conv2d", _C2,
                 f"MMA contraction extent K + {_d} not divisible by the tensor-core atom (16)",
                 (_C2_CALL, _C2_CALL.replace("Cout/#qq, K);", f"Cout/#qq, K + {_d});"))))
M3.append(_m("M3.s2.cv2.b1", "M3", 2, "stride", "conv2d", _C2,
             "misaligned tensor-core base: shared weight chunk offset by 1",
             ("l1_B = dma.copy B.chunkat(qq, _) => local;",
              "l1_B = dma.copy B.chunkat(qq + 1, _) => local;")))
M3.append(_m("M3.s2.cv2.i1", "M3", 2, "stride", "conv2d", _C2,
             "misaligned DMA base address on the input tensor",
             ("i.chunkat(p#n, _, _, _).span_as(K, Ho, Wo)",
              "i.chunkat(p#n + 1, _, _, _).span_as(K, Ho, Wo)")))
for _mul in (8, 16):
    M3.append(_m(f"M3.s3.cv2.local{_mul}", "M3", 3, "dim-mismatch", "conv2d", _C2,
                 f"local accumulator tile {_mul}x over the per-thread budget",
                 ("local f32 [M/#q, Cout/#qq] l1_Y{0.0f};",
                  "local f32 [M/#q, Cout/#qq * %d] l1_Y{0.0f};" % _mul)))
for _cin in (512, 1024):
    # 32 * Cin * 4 B = 64 KB / 128 KB in shared, over the 48 KB limit.
    M3.append(_m(f"M3.s3.cv2.shared{_cin}", "M3", 3, "dim-mismatch", "conv2d", _C2,
                 f"full weight matrix 32 x {_cin} f32 = {32 * _cin * 4 // 1024} KB held in "
                 f"shared, over the 48 KB device limit",
                 ("f32 [32, 128, 1, 1] w", f"f32 [32, {_cin}, 1, 1] w"),
                 ("auto w = choreo::make_spandata<choreo::f32>(32, 128, 1, 1);",
                  f"auto w = choreo::make_spandata<choreo::f32>(32, {_cin}, 1, 1);")))

ALL = {"M1": M1, "M2": M2, "M3": M3}


def transforms_for(cls):
    return ALL[cls]


def categories_for(cls, level2=False):
    """Categories this class is generated on, at the requested widening level.

    Starts from the declared minimal / level-2 sets (specs §5) and unions in any
    category a REGISTERED spec declares as required, so a spec that needs a
    category outside its class's default set (M2.16 lives on `conv2d`) is not
    silently dropped by selection.
    """
    cats = list(MINIMAL_SET[cls])
    if level2:
        cats += [c for c in LEVEL2_SET.get(cls, []) if c not in cats]
    for m in ALL.get(cls, []):
        if m.category not in cats and LEVEL2_SET.get(cls) and level2:
            cats.append(m.category)
    return cats


# ===========================================================================
# v2.1 additions (specs §1-§4, §9.6.2)
#
# Every operator below anchors on source text that a v1 operator already
# matched, so GATE 2 is a check rather than a hope. Budget follows the path
# (§9.6.1): P1 specs get several operators per category so the -rtc curve has
# something to sweep; P3/P4 specs get ONE per (spec x category) cell, because
# no obligation exists at any threshold.
# ===========================================================================

# ---- M1.7 reversed loop bound -> overrun, not empty -----------------------
M1 += [
    _m("M1.7.rl1.revbound", "M1", 7, "stride", "relu",
       "1_bert_32x512x768_32x512x768",
       "reversed loop extent: the outer takes the innermost bound, so the "
       "tile index overruns instead of the tail being correctly empty",
       ("foreach {i, j, k} in [I / #p, J, 12]",
        "foreach {i, j, k} in [12, J, I / #p]")),
    _m("M1.7.tp11.revbound", "M1", 7, "stride", "transpose",
       "11_dynamic_32xSx768_32x768xS",
       "reversed loop extent on the dynamic inner tile",
       ("foreach {y, z} in [seq_len, 12]", "foreach {y, z} in [12, seq_len]")),
]

# ---- M1.8 stride scaling (stride x 2) -------------------------------------
M1 += [
    _m("M1.8.rl1.stride2", "M1", 8, "stride", "relu",
       "1_bert_32x512x768_32x512x768",
       "stride scaling: the tile coordinate advances by 2, skipping every "
       "other tile and reading a neighbouring tile's rows",
       ("inp.chunkat(p#i, j, k, _)", "inp.chunkat(p#i, j, k * 2, _)")),
    _m("M1.8.tp1.stride2", "M1", 8, "stride", "transpose",
       "1_bert_32x512x768_32x768x512",
       "stride scaling: the source tile coordinate advances by 2",
       ("i.chunkat(p, y, z)", "i.chunkat(p, y * 2, z)")),
    _m("M1.8.ln1.stride2", "M1", 8, "stride", "layer_normalization",
       "1_bert_32x512x768_768_768",
       "stride scaling: the row index advances by 2 within the chunk",
       ("lhs.at(p#n, j, k)", "lhs.at(p#n, j * 2, k)")),
]

# ---- M1.9 base offset applied WITHOUT shrinking the extent ----------------
# v1 spec 5 ("offset view") conflated two realizations. The one that moves the
# BASE and leaves the declared extent alone is M1.9; the one that moves an
# element index inside the tile is M1.5.
M1 += [
    _m("M1.9.rl1.baseoff", "M1", 9, "oob", "relu",
       "1_bert_32x512x768_32x512x768",
       "base offset without shrinking the extent: the innermost tile origin "
       "moves by one, so the last element reads past the tile",
       ("inp.chunkat(p#i, j, k, _)", "inp.chunkat(p#i, j, k + 1, _)")),
    _m("M1.9.sm1.baseoff", "M1", 9, "oob", "softmax",
       "1_bert_32x512x768_32x512x768",
       "base offset without shrinking the extent on the reduction read",
       ("l1_input.data.at(0, j, k)", "l1_input.data.at(0, j, k + 1)")),
    _m("M1.9.tp1.baseoff", "M1", 9, "oob", "transpose",
       "1_bert_32x512x768_32x768x512",
       "base offset without shrinking the extent on the transpose source",
       ("i.chunkat(p, y, z)", "i.chunkat(p, y, z + 1)")),
]

# ---- M1.10 tile-boundary rounding (floor vs ceil on the last tile) --------
M1 += [
    _m("M1.10.rl1.ceil", "M1", 10, "oob", "relu",
       "1_bert_32x512x768_32x512x768",
       "tile-boundary rounding: one extra outer tile is taken, so the last "
       "block starts past the extent (ceil instead of floor)",
       ("foreach {i, j, k} in [I / #p, J, 12]",
        "foreach {i, j, k} in [I / #p + 1, J, 12]")),
    _m("M1.10.rl11.ceil", "M1", 10, "oob", "relu",
       "11_dynamic_32xSx768_32xSx768",
       "tile-boundary rounding on the dynamic outer extent",
       ("foreach {i, j, k} in [I / #p, NUM_HEADS, 12]",
        "foreach {i, j, k} in [I / #p + 1, NUM_HEADS, 12]")),
]

# ---- M1.11 read-after-write aliasing overlap ------------------------------
M1 += [
    _m("M1.11.rl1.alias", "M1", 11, "stride", "relu",
       "1_bert_32x512x768_32x512x768",
       "read-after-write aliasing: the shared tile is shrunk below the "
       "`parallel q by 64` fan-out, so threads overwrite each other's region",
       ("shared f32 [1, 1, 64, 1] inp_s, out_s;",
        "shared f32 [1, 1, 32, 1] inp_s, out_s;")),
    _m("M1.11.tp1.alias", "M1", 11, "stride", "transpose",
       "1_bert_32x512x768_32x768x512",
       "read-after-write aliasing on the transpose source tile",
       ("shared f32 [1, 1, 64] is;", "shared f32 [1, 1, 32] is;")),
]

# ---- M1.12 wrong loop variable for a dimension (broadcast index reuse) ----
M1 += [
    _m("M1.12.sm1.idxreuse", "M1", 12, "stride", "softmax",
       "1_bert_32x512x768_32x512x768",
       "wrong loop variable for a dimension: the reduction index is reused "
       "for the row dimension (broadcast index reuse)",
       ("l1_input.data.at(0, j, k)", "l1_input.data.at(0, k, k)")),
    _m("M1.12.ln1.idxreuse", "M1", 12, "stride", "layer_normalization",
       "1_bert_32x512x768_768_768",
       "wrong loop variable for a dimension on the primary operand",
       ("lhs.at(p#n, j, k)", "lhs.at(p#n, j, j)")),
]

# ---- M1.13 symbolic-bound overrun ----------------------------------------
M1 += [
    _m("M1.13.tp11.symbound", "M1", 13, "oob", "transpose",
       "11_dynamic_32xSx768_32x768xS",
       "symbolic-bound overrun: the index runs one past a bound expressed "
       "over the runtime parameter seq_len",
       ("foreach {y, z} in [seq_len, 12]", "foreach {y, z} in [seq_len + 1, 12]")),
    _m("M1.13.rl11.symbound", "M1", 13, "oob", "relu",
       "11_dynamic_32xSx768_32xSx768",
       "symbolic-bound overrun on the runtime head count",
       ("foreach {i, j, k} in [I / #p, NUM_HEADS, 12]",
        "foreach {i, j, k} in [I / #p, NUM_HEADS + 1, 12]")),
]

# ---- M1.14 chunkat tile-coordinate over/underflow -------------------------
# The load-bearing v2.1 addition: a check the compiler WROTE and then DISABLED
# (0 of 272 chunkat obligations enabled at -rtc=entry). v1 filed these under
# "offset view"; the coordinate that moves is the TILE coordinate, not an
# element index, so they belong to M1.14.
M1 += [
    _m("M1.14.sm1.store", "M1", 14, "oob", "softmax",
       "1_bert_32x512x768_32x512x768",
       "tile-coordinate overrun on the write-back chunk while the element "
       "index inside the tile stays in bounds",
       ("dma.copy l1_out => output.chunkat(i#p, q, _)",
        "dma.copy l1_out => output.chunkat(i#p + 1, q, _)")),
    _m("M1.14.sm11.store", "M1", 14, "oob", "softmax",
       "11_dynamic_32xSx768_32xSx768",
       "tile-coordinate overrun on the dynamic write-back chunk",
       ("dma.copy l1_out => output.chunkat(i#p, _, _)",
        "dma.copy l1_out => output.chunkat(i#p + 1, _, _)")),
    _m("M1.14.rl1.load", "M1", 14, "oob", "relu",
       "1_bert_32x512x768_32x512x768",
       "tile-coordinate overrun on the async DMA source chunk",
       ("inp.chunkat(p#i, j, k, _)", "inp.chunkat(p#i + 1, j, k, _)")),
    _m("M1.14.rl11.store", "M1", 14, "oob", "relu",
       "11_dynamic_32xSx768_32xSx768",
       "tile-coordinate overrun on the DMA destination chunk",
       ("dma.copy out_s => out.chunkat(p#i, j, k, _)",
        "dma.copy out_s => out.chunkat(p#i + 1, j, k, _)")),
    _m("M1.14.tp1.load", "M1", 14, "oob", "transpose",
       "1_bert_32x512x768_32x768x512",
       "tile-coordinate overrun on the transpose source chunk",
       ("i.chunkat(p, y, z)", "i.chunkat(p + 1, y, z)")),
]

# ---- M1.15 / M1.16 / M1.17 / M1.18: screened, no operator yet -------------
# M1.15 (dimof rank)   -- the dimof mechanism emits 0 runtime obligations; the
#                         spec's whole point is that a miss is the finding.
# M1.16 (select factor)-- needs a `select` category; the no-static-factor path
#                         is unassessed (defect F10).
# M1.17 (5th index)    -- needs a rank-5 category.
# M1.18 (interval vs canonical) -- a mechanism-level claim, not an edit shape;
#                         expressed by re-running an existing mutant under the
#                         other mechanism once the ledger carries both.
# These are registry `pending`, so `gen_mutants.py --report` lists them.

# ---- M1.19 index-carrier overflow (narrow-carrier, realization b) ---------
# A single dimension in (2^31, 2^32): LEGAL under `__inf__ = 2^32 - 1`, but a
# 32-bit signed carrier cannot name an index into it (cute_codegen.cpp:1757
# emits `(int)(<coord> * <extent>)`). No allocation: the extent is DECLARED,
# not materialised. A surviving mutant is unambiguously a compiler defect.
_INT32_MAX_EXTENT = "2147483648"
M1 += [
    _m("M1.19.ln1.carrier.scale", "M1", 19, "oob", "layer_normalization",
       "1_bert_32x512x768_768_768",
       "index-carrier overflow: a legal extent in (2^31, 2^32) that a 32-bit "
       "signed carrier cannot name",
       ("f32 [K] scale", "f32 [%s] scale" % _INT32_MAX_EXTENT)),
    _m("M1.19.ln3.carrier.bias", "M1", 19, "oob", "layer_normalization",
       "3_attention_32xNx512x64_64_64",
       "index-carrier overflow on the dynamic-case secondary operand",
       ("f32 [L] bias", "f32 [%s] bias" % _INT32_MAX_EXTENT)),
]

# ---- M1.20 view / subspan offset, stride, rank arity (StaticFail-only) ----
# _StaticFail_ is a COUNTER, not an assessment creator (shapeinfer.cpp:36-52):
# not-statically-known -> nothing at all. So a SYMBOLIC arity change is silent
# at every -rtc. P3 -> one injection per cell.
M1 += [
    _m("M1.20.tp1.arity", "M1", 20, "wrong-shape", "transpose",
       "1_bert_32x512x768_32x768x512",
       "view rank arity changed from 3 to 4 with a symbolic length, so "
       "_StaticFail_ cannot resolve it and nothing is emitted",
       ("f32 [i.span(0), i.span(2), i.span(1)] o;",
        "f32 [i.span(0), i.span(2), i.span(1), 1] o;")),
    _m("M1.20.tp11.arity", "M1", 20, "wrong-shape", "transpose",
       "11_dynamic_32xSx768_32x768xS",
       "view rank arity changed on the dynamic transpose output",
       ("f32 [i.span(0), i.span(2), i.span(1)] o;",
        "f32 [i.span(0), i.span(2), i.span(1), 1] o;")),
]

# ---- M1.21 tileat/at index vs the tiled extent ----------------------------
M1 += [
    _m("M1.21.tp1.extent", "M1", 21, "oob", "transpose",
       "1_bert_32x512x768_32x768x512",
       "tile index vs tiled extent on a non-divisible extent: 513 rows with a "
       "12-wide tile leaves a tail whose element index exceeds the declared 512",
       ("foreach {y, z} in [512, 12]", "foreach {y, z} in [513, 12]")),
    _m("M1.21.ln1.tileat", "M1", 21, "oob", "layer_normalization",
       "1_bert_32x512x768_768_768",
       "tile index one past the tiled extent on the primary operand",
       ("lhs.at(p#n, j, k)", "lhs.at(p#n, j, k + K)")),
]

# ---- M2.6 two extents transposed -----------------------------------------
M2 += [
    _m("M2.6.mm1.extswap", "M2", 6, "dim-mismatch", "matmul",
       "1_bert_32x512x768_768x768_32x512x768",
       "two leading extents transposed in the output declaration",
       ("f32 [lhs.span(0), lhs.span(1), rhs.span(1)] output;",
        "f32 [lhs.span(1), lhs.span(0), rhs.span(1)] output;")),
    _m("M2.6.cc1.extswap", "M2", 6, "dim-mismatch", "concat",
       "1_bert_32x512x768_32x512x768_32x512x1536",
       "two extents transposed on the first concat operand",
       ("f32 [I, J, K, L] a,", "f32 [J, I, K, L] a,")),
]

# ---- M2.7 reduced-rank view (a dimension dropped) -------------------------
M2 += [
    _m("M2.7.mm1.rank", "M2", 7, "dim-mismatch", "matmul",
       "1_bert_32x512x768_768x768_32x512x768",
       "reduced-rank view: a dimension is dropped from the output declaration",
       ("f32 [lhs.span(0), lhs.span(1), rhs.span(1)] output;",
        "f32 [lhs.span(0), rhs.span(1)] output;")),
    _m("M2.7.cc1.rank", "M2", 7, "dim-mismatch", "concat",
       "1_bert_32x512x768_32x512x768_32x512x1536",
       "reduced-rank view on the second concat operand",
       ("f32 [I, J, K, L] b)", "f32 [I, J, K] b)")),
]

# ---- M2.8 broadcast extent set to 1 instead of N --------------------------
M2 += [
    _m("M2.8.ln1.bcast1", "M2", 8, "dim-mismatch", "layer_normalization",
       "1_bert_32x512x768_768_768",
       "broadcast extent set to 1 instead of K on the secondary operand",
       ("f32 [K] bias", "f32 [1] bias")),
    _m("M2.8.ln1.caller.bcast1", "M2", 8, "dim-mismatch", "layer_normalization",
       "1_bert_32x512x768_768_768",
       "broadcast extent set to 1 in the caller's materialisation",
       ("auto bias = choreo::make_spandata<choreo::f32>(K);",
        "auto bias = choreo::make_spandata<choreo::f32>(1);")),
]

# ---- M2.9 batch/group dimension swapped -----------------------------------
M2 += [
    _m("M2.9.ln3.batchswap", "M2", 9, "dim-mismatch", "layer_normalization",
       "3_attention_32xNx512x64_64_64",
       "batch/group dimension swapped on the primary operand",
       ("f32 [I, N0, K, L] lhs", "f32 [N0, I, K, L] lhs")),
    _m("M2.9.cc11.batchswap", "M2", 9, "dim-mismatch", "concat",
       "11_dynamic_32xS1x768_32xS2x768_32xS1pS2x768",
       "batch/group dimension swapped on the first dynamic concat operand",
       ("f32 [I, J1, K, L] a,", "f32 [J1, I, K, L] a,")),
]

# ---- M2.10 transpose permutation on a SQUARE operand ----------------------
# extents stay equal, so semacheck.cpp:1073's extent-only comparison passes;
# the memory order changes anyway. The M2 flagship of the view family.
M2 += [
    _m("M2.10.mm1.square", "M2", 10, "wrong-shape", "matmul",
       "1_bert_32x512x768_768x768_32x512x768",
       "square transpose: permutation and tile coordinates move TOGETHER, so "
       "every extent agrees and only the layout is wrong",
       ("dma.copy l1_out => output.chunkat(p#q, m_tile, n_tile);",
        "dma.transp<1,0> l1_out => output.chunkat(p#q, n_tile, m_tile);")),
    _m("M2.10.mm11.square", "M2", 10, "wrong-shape", "matmul",
       "11_dynamic_32xSx768_768x768_32xSx768",
       "square transpose on the dynamic operand pair",
       ("dma.copy l1_out => output.chunkat(p#q, m_tile, n_tile);",
        "dma.transp<1,0> l1_out => output.chunkat(p#q, n_tile, m_tile);")),
]

# ---- M2.11 DMA to-buffer element-count undersize --------------------------
M2 += [
    _m("M2.11.rl1.undersize", "M2", 11, "wrong-shape", "relu",
       "1_bert_32x512x768_32x512x768",
       "DMA destination buffer undersized by one element on an otherwise "
       "LogicalEqual path",
       ("shared f32 [1, 1, 64, 1] inp_s, out_s;",
        "shared f32 [1, 1, 63, 1] inp_s, out_s;")),
    _m("M2.11.cc1.undersize", "M2", 11, "wrong-shape", "concat",
       "1_bert_32x512x768_32x512x768_32x512x1536",
       "shared source tile undersized by one element",
       ("shared f32 [1, 1, K, L] as, bs;", "shared f32 [1, 1, K - 1, L] as, bs;")),
]

# ---- M2.12 rank mismatch through .pad -- NOT APPLICABLE (P2, repaired) ----
# semacheck.cpp:1081-1090 is a hard Error1 that produces no ledger row at all.
# Recorded in SPEC_REGISTRY with prohibition "repaired"; no operator exists and
# none may be written (specs §9.5.0). This is a positive verdict, not a gap.

# ---- M2.13 shape-equal / layout-unequal -----------------------------------
# every extent agrees; the affine map does not. No -rtc value reaches this.
M2 += [
    _m("M2.13.mm1.affine", "M2", 13, "wrong-shape", "matmul",
       "1_bert_32x512x768_768x768_32x512x768",
       "shape-equal / layout-unequal: the rhs tile coordinates are exchanged "
       "on a SQUARE operand, so no extent disagrees",
       ("l1_b = dma.transp<1,0> rhs.chunkat(k_tile, n_tile) => local;",
        "l1_b = dma.transp<1,0> rhs.chunkat(n_tile, k_tile) => local;")),
    _m("M2.13.mm11.affine", "M2", 13, "wrong-shape", "matmul",
       "11_dynamic_32xSx768_768x768_32xSx768",
       "shape-equal / layout-unequal on the dynamic operand pair",
       ("l1_b = dma.transp<1,0> rhs.chunkat(k_tile, n_tile) => local;",
        "l1_b = dma.transp<1,0> rhs.chunkat(n_tile, k_tile) => local;")),
]

# ---- M2.14 contraction-dim mismatch masked by a square operand ------------
M2 += [
    _m("M2.14.mm1.maskedK", "M2", 14, "dim-mismatch", "matmul",
       "1_bert_32x512x768_768x768_32x512x768",
       "contraction extent taken from the wrong operand's dimension; equal by "
       "construction because the operand is square, so the mismatch is masked",
       (_MM1_CALL, _MM1_CALL.replace("l1_a.span(2));", "l1_b.span(1));"))),
]

# ---- M2.16 span_as preserving ElementCount with a different split ---------
# semacheck.cpp:947 compares COUNT, not shape.
M2 += [
    _m("M2.16.cv1.countonly", "M2", 16, "dim-mismatch", "conv2d",
       _C1,
       "span_as split exchanged: ElementCount is preserved, so the count-only "
       "comparison at semacheck.cpp:947 passes",
       ("dma.copy w.span_as(Cout, K).chunkat(_, kt) => B_tile;",
        "dma.copy w.span_as(K, Cout).chunkat(_, kt) => B_tile;")),
]

# ---- M2.17 span_as / reshape on runtime-shaped data (check skipped) -------
# semacheck.cpp:946-950 is guarded by !RuntimeShaped() on BOTH sides, so a
# symbolic operand escapes the size check entirely. P3 -> one per cell.
M2 += [
    _m("M2.17.ln3.rtshaped", "M2", 17, "dim-mismatch", "layer_normalization",
       "3_attention_32xNx512x64_64_64",
       "runtime-shaped operand: the size check is skipped entirely because "
       "one side is not statically shaped",
       ("f32 [I, N0, K, L] lhs", "f32 [I, N0, K, L + 1] lhs")),
    _m("M2.17.cc11.rtshaped", "M2", 17, "dim-mismatch", "concat",
       "11_dynamic_32xS1x768_32xS2x768_32xS1pS2x768",
       "runtime-shaped concat operand escapes the size check",
       ("f32 [I, J_OUT, K, L] out;", "f32 [I, J_OUT, K + 1, L] out;")),
]

# ---- M2.18 reshape on a non-contiguous span -- WARNING ONLY (P4) ----------
# semacheck.cpp:1195+ emits `Warning(rop->LOC(), ...)` and continues. P4 specs
# need one injection per cell, and the finding is the warning, not the miss.

# ---- M2.19 DMA extent mismatch with a symbolic extent ---------------------
# semacheck.cpp:1140-1145 forces emit_error false when either extent is
# symbolic: the hard error is escaped. P3.
M2 += [
    _m("M2.19.mm11.symdma", "M2", 19, "dim-mismatch", "matmul",
       "11_dynamic_32xSx768_768x768_32xSx768",
       "DMA source/destination disagreement where the extent is symbolic, so "
       "the hard error at semacheck.cpp:1140-1145 is escaped",
       ("dma.copy l1_out => output.chunkat(p#q, m_tile, n_tile);",
        "dma.copy l1_out => output.chunkat(p#q, m_tile + 1, n_tile);")),
    _m("M2.19.cc11.symdma", "M2", 19, "dim-mismatch", "concat",
       "11_dynamic_32xS1x768_32xS2x768_32xS1pS2x768",
       "shared destination tile disagrees with a symbolic DMA source extent",
       ("shared f32 [1, J_OUT, 1, 1] os;", "shared f32 [1, J_OUT, 1, 2] os;")),
]

# ---- M3.14 linear .copy with a dim >= 2^24 -- the check is ABSENT ---------
# gpu_adapt.hpp:320 `// linear copy` ... `// omitted`. The other six cells of
# the DMA matrix call CheckDimSize; this one does not (defect F1). P3.
M3 += [
    _m("M3.14.mm1.linearcopy", "M3", 14, "dim-mismatch", "matmul",
       "1_bert_32x512x768_768x768_32x512x768",
       "linear .copy with a dimension >= 2^24 reached by STRIDE, on the one "
       "DMA-matrix cell that carries no CheckDimSize call",
       ("l1_a = dma.copy lhs.chunkat(p#q, m_tile, k_tile) => local;",
        "l1_a = dma.copy lhs.chunkat(p#q, m_tile, 16777216) => local;")),
]

# ---- M3.15 .pad with a dim >= 2^24 -- the pad path never checks -----------
# gpu_adapt.hpp:360-450: RankLE5, pad ranges and padding_mid only (defect F2).
M3 += [
    _m("M3.15.cv1.pad2p24", "M3", 15, "dim-mismatch", "conv2d",
       _C1,
       "shared tile beyond 2^24 elements on the pad path, which never calls "
       "CheckDimSize",
       ("shared f32 [Cout, 8] B_tile;", "shared f32 [Cout, 16777216] B_tile;")),
]

# ---- M3.16 TMA box inner alignment with a SYMBOLIC leading dim ------------
# gpu_adapt.hpp:640 `// TODO: emit runtime assessment` -- no check at all
# (defect F3). P3.
M3 += [
    _m("M3.16.cv1.symbox", "M3", 16, "dim-mismatch", "conv2d",
       _C1,
       "TMA box inner alignment with a symbolic leading dim: the assessment "
       "was never emitted, so no threshold reaches it",
       ("i.chunkat(p#n, _, _, _).span_as(K, Ho, Wo)",
        "i.chunkat(p#n, _, _, _).span_as(K, Ho + 1, Wo)")),
]

# ---- M3.9 pad-field overrun ---------------------------------------------
# gpu_adapt.hpp:360-450 assesses `RankLE5`, the per-dim pad ranges and the
# `padding_mid` vector -- and nothing else. The pad FIELDS are therefore on an
# assessed path (P1), unlike M3.14/M3.15 which sit on the unchecked 2^24
# magnitude path. The overrun is bounded (+1 element on one dim) so the padded
# tile stays the same size class and the mutant never allocates: what moves is
# the *placement*, which is value-observable through the im2col index map.
_PAD10 = ("dma.pad<{0, 0, padding, padding}, {0, 0, padding, padding}, "
          "{0, 0, 0, 0}, 0.0f> i.chunkat(p#n, _, _, _) => shared;")
M3 += [
    _m("M3.9.cv10.padhigh", "M3", 9, "dim-mismatch", "conv2d",
       "10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D",
       "pad_high on the innermost dim exceeds the assessed range for that dim "
       "by one element: every im2col window shifts by one column",
       (_PAD10, _PAD10.replace("{0, 0, padding, padding}, {0, 0, 0, 0}",
                               "{0, 0, padding, padding + 1}, {0, 0, 0, 0}"))),
    _m("M3.9.cv10.padlow", "M3", 9, "dim-mismatch", "conv2d",
       "10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D",
       "pad_low on the out-of-channel dim exceeds the assessed range by one "
       "element: the padded rows shift, so the halo the stencil reads is the "
       "wrong halo",
       (_PAD10, _PAD10.replace("{0, 0, padding, padding},",
                               "{0, 0, padding + 1, padding},", 1))),
]

# ---- M3.10 mid-padding on a dim that must not carry it ------------------
# `padding_mid[rank-1] == 0` is the obligation; a non-zero entry means the
# hardware re-orders the tile against the compiler's own index map. The insert
# is in-place (padding_mid repositions content, it does not grow the tile), so
# this is the one M3 family that is value-observable at zero allocation cost.
M3 += [
    _m("M3.10.cv10.midlast", "M3", 10, "dim-mismatch", "conv2d",
       "10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D",
       "padding_mid non-zero on the innermost dim, violating "
       "padding_mid[rank-1] == 0: the tile arrives re-ordered",
       (_PAD10, _PAD10.replace("{0, 0, 0, 0}, 0.0f>", "{0, 0, 0, 1}, 0.0f>"))),
    _m("M3.10.cv10.midinner", "M3", 10, "dim-mismatch", "conv2d",
       "10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D",
       "padding_mid non-zero on an interior dim: the mid-pad vector is "
       "assessed but its entries are not constrained to be zero outside the "
       "innermost position",
       (_PAD10, _PAD10.replace("{0, 0, 0, 0}, 0.0f>", "{0, 0, 1, 0}, 0.0f>"))),
]

# ===========================================================================
# M4 -- control-flow / iteration-space (specs §4)
#
# These are not memory-safety mutants: they ask whether the compiler reasons
# about the ITERATION SPACE at all. A zero trip count is legal (the body simply
# never executes), so a "miss" here is the strongest possible finding -- the
# compiler cannot be reasoning about the loop at all.
# ===========================================================================

M4 = []
M4 += [
    _m("M4.1.cv10.zeroextent", "M4", 1, "stride", "conv2d",
       "10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D",
       "with-in mdspan extent mutated to 0: the loop body cannot execute, so "
       "any assessed obligation in it is vacuous",
       ("with {i, j} in [Hpad/2, Wpad/2]", "with {i, j} in [0, Wpad/2]")),
    _m("M4.1.rl1.zeroextent", "M4", 1, "stride", "relu",
       "1_bert_32x512x768_32x512x768",
       "with-in mdspan extent mutated to 0 on the element loop",
       ("foreach {i, j, k} in [I / #p, J, 12]",
        "foreach {i, j, k} in [0, J, 12]")),
]

M4 += [
    _m("M4.2.cv10.zero", "M4", 2, "stride", "conv2d",
       "10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D",
       "parallelby bound mutated to 0: a legal empty iteration space that "
       "still carries the whole body's obligations",
       ("parallel q by 8  {", "parallel q by 0  {")),
    _m("M4.2.cv1.negative", "M4", 2, "stride", "conv2d",
       _C1, "parallelby bound mutated to negative",
       ("parallel p by 2  {", "parallel p by -2  {")),
]

M4 += [
    _m("M4.3.cv10.symzero", "M4", 3, "stride", "conv2d",
       "10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D",
       "parallelby bound symbolic and zero only at runtime: statically "
       "indistinguishable from a real loop, so only a runtime assessment "
       "can catch it",
       ("foreach n in [N / #p]", "foreach n in [N - N]")),
    _m("M4.3.tp1.symzero", "M4", 3, "stride", "transpose",
       "1_bert_32x512x768_32x768x512",
       "symbolic loop bound that vanishes at runtime on the transpose path",
       ("foreach {y, z} in [512, 12]", "foreach {y, z} in [512 - 512, 12]")),
]

# M4.4 (bound > 0 but the iteration space is empty) -- NOOP BY CONSTRUCTION.
# Registered `admissible=False` (specs §4): included only as the negative
# control that calibrates the noop rate. No operator: an edit producing that
# state would be a generation-side duplicate of M4.1/M4.2.

M4 += [
    _m("M4.5.cv10.zerostep", "M4", 5, "stride", "conv2d",
       "10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D",
       "zero stride in the index map: every iteration reads tile 0, so the "
       "write-back overwrites a single tile N times",
       ("dma.copy l1_Y => Y.chunkat(q#_q, qq);",
        "dma.copy l1_Y => Y.chunkat(q#_q, qq * 0);")),
]

# ---- M2.15 pad_low <-> pad_high swapped (length preserved) ---------------
# The check at semacheck.cpp:1076-1100 SUMs the pad fields, so it is blind to
# placement. Exchanging pad_low and pad_high per axis leaves the total (and
# thus the padded length) unchanged while moving every element.
M2 += [
    _m("M2.15.cv10.padswap", "M2", 15, "wrong-shape", "conv2d",
       "10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D",
       "pad_low and pad_high swapped per axis: the padded length is unchanged, "
       "so the sum-based check cannot see it, but every element moves",
       ("dma.pad<{0, 0, padding, padding}, {0, 0, padding, padding}, "
        "{0, 0, 0, 0}, 0.0f> i.chunkat(p#n, _, _, _) => shared;",
        "dma.pad<{0, padding, padding, 0}, {0, padding, padding, 0}, "
        "{0, 0, 0, 0}, 0.0f> i.chunkat(p#n, _, _, _) => shared;")),
    _m("M2.15.cv4.padswap", "M2", 15, "wrong-shape", "conv2d",
       "4_dynamic_32x128xHxW_256x128x3x3_32x256xHxW_S_P_D",
       "pad_low and pad_high swapped on the per-thread padded tile",
       ("dma.pad<{0, 0, padding, padding}, {0, 0, padding, padding}, "
        "{0, 0, 0, 0}, 0.0f> i.chunkat(p#q#n, ci, _, _) => local;",
        "dma.pad<{0, padding, padding, 0}, {0, padding, padding, 0}, "
        "{0, 0, 0, 0}, 0.0f> i.chunkat(p#q#n, ci, _, _) => local;")),
]

# ---- M2.21 MSB broadcast extent neither 1 nor equal ----------------------
# semacheck.cpp:466-500 walks the TRAILING dimensions only, so a disagreement
# in the leading (MSB) dimension of a rank-unequal operand pair is invisible.
M2 += [
    _m("M2.21.rl1.msbbcast", "M2", 21, "dim-mismatch", "relu",
       "1_bert_32x512x768_32x512x768",
       "MSB broadcast extent neither 1 nor equal: the rank-unequal path walks "
       "trailing dimensions only, so the leading disagreement is invisible",
       ("shared f32 [1, 1, 64, 1] inp_s, out_s;",
        "shared f32 [2, 1, 64, 1] inp_s, out_s;")),
    _m("M2.21.rl11.msbbcast", "M2", 21, "dim-mismatch", "relu",
       "11_dynamic_32xSx768_32xSx768",
       "MSB broadcast disagreement on the symbolic (dynamic-shape) case: the "
       "same trailing-only walk, but every extent is a runtime value, so the "
       "static screen has nothing to compare and the violation reaches the "
       "device unchecked",
       ("shared f32 [1, 1, 64, 1] inp_s, out_s;",
        "shared f32 [2, 1, 64, 1] inp_s, out_s;")),
]

# ---- L launch status: inadmissible by construction -----------------------
# One operator for the strongest cell (L4, block extent not a multiple of 32)
# so the class is exercised end to end; the rest stay `pending` and are listed
# by the GATE 2 cross-check. `spec_id` must be explicit: the class letter is
# not the spec-number prefix.
L = []
L += [
    _m("L4.cv1.block3", "L", 4, "hw", "conv2d",
       _C1,
       "block extent not a multiple of 32: a launch-status failure, not an "
       "obligation -- exercises the observation channel and is excluded from "
       "every admissible denominator",
       ("parallel p by 2  {", "parallel p by 3  {"), spec_id="L4"),
]

ALL["M4"] = M4
ALL["L"] = L


# ===========================================================================
# GATE 2 -- reconcile the registry against the operators that actually loaded.
#
# Doing this in code rather than by hand is the whole point: a hand-written
# `status=` drifts the moment an operator is added or removed, and a drifted
# registry is worse than no registry because it reads as coverage. After this
# block, SPEC_REGISTRY[sid]["status"] is exactly one of:
#
#   "implemented"  >=1 operator loaded for this spec
#   "na"           not applicable (P2, or declared inadmissible) -- recorded,
#                  never generated (specs §9.5.0). A positive verdict.
#   "pending"      applicable and screened, operator not written yet
#
# and `operator_specs` / `pending_specs` / `na_specs` are the derived views
# `gen_mutants.py` reports and `collect.py` stamps onto every record.
# ===========================================================================

operator_specs = set()
for _cls in sorted(ALL):
    for _m_ in ALL[_cls]:
        operator_specs.add(_m_.spec_id)

_unregistered = sorted(s for s in operator_specs if s not in SPEC_REGISTRY)
if _unregistered:
    raise ValueError(
        "operator(s) claim spec ids absent from SPEC_REGISTRY: %s"
        % ", ".join(_unregistered))

_PROHIBITIONS = ("absent", "derived", "repaired", "harness-owned", "observation")

for _sid, _meta in SPEC_REGISTRY.items():
    # -- axis 1: does an operator exist? ---------------------------------
    _meta["status"] = "implemented" if _sid in operator_specs else "pending"

    # -- axis 2: may the spec enter the admissible denominator? -----------
    # P2 (the compiler repairs the state) and the noop controls are
    # inadmissible *by classification*, whether or not an operator exists --
    # which is why this cannot be folded into `status`: M1.6/M3.6/M4.4/L1/L2/L4
    # are generated on purpose.
    if _meta["path"] == "P2" or _sid in NOOP_BY_CONSTRUCTION:
        _meta["admissible"] = False

    if not _meta["admissible"]:
        # An inadmissible spec that does not say WHY is a silent gap. Refuse it,
        # exactly as the unregistered-operator check above does.
        if _meta["prohibition"] not in _PROHIBITIONS:
            raise ValueError(
                "spec %s is inadmissible but gives no valid prohibition "
                "(got %r, want one of %s) -- record the reason or restore "
                "admissibility"
                % (_sid, _meta["prohibition"], ", ".join(_PROHIBITIONS)))
    elif _meta["prohibition"]:
        raise ValueError(
            "spec %s is admissible but declares a prohibition (%r)"
            % (_sid, _meta["prohibition"]))

pending_specs = sorted(s for s, v in SPEC_REGISTRY.items()
                       if v["status"] == "pending" and v["admissible"])
unwritten_specs = sorted(s for s, v in SPEC_REGISTRY.items()
                         if v["status"] == "pending")
na_specs = sorted(s for s, v in SPEC_REGISTRY.items()
                  if not v["admissible"])


def _admissible_specs():
    return sorted(s for s, v in SPEC_REGISTRY.items() if v["admissible"])


def registry_summary():
    """The GATE 2 report: coverage, admissibility, and what is still open.

    Reports both axes so that neither can be misread:
      status      implemented | pending   -- does an operator exist?
      admissible  True | False            -- may it enter the denominator?
      prohibition reason, for the inadmissible ones
    """
    import collections as _c
    na_by_reason = _c.OrderedDict()
    for _s, _v in sorted(SPEC_REGISTRY.items()):
        if not _v["admissible"]:
            na_by_reason.setdefault(_v["prohibition"], []).append(_s)
    return {
        "n_specs": len(SPEC_REGISTRY),
        "status_counts": dict(_c.Counter(v["status"]
                                         for v in SPEC_REGISTRY.values())),
        "admissible_counts": dict(_c.Counter(bool(v["admissible"])
                                              for v in SPEC_REGISTRY.values())),
        "prohibition_counts": dict(_c.Counter(
            v["prohibition"] for v in SPEC_REGISTRY.values()
            if v["prohibition"])),
        "path_counts": dict(_c.Counter(v["path"]
                                       for v in SPEC_REGISTRY.values())),
        "implemented": sorted(operator_specs),
        "admissible": _admissible_specs(),
        "pending": pending_specs,
        "unwritten": unwritten_specs,
        "na": na_specs,
        "na_by_reason": na_by_reason,
        "n_operators": {c: len(ALL[c]) for c in sorted(ALL)},
    }
