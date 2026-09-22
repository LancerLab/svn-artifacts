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
  * 4 classes: M1 element-access, M2 shape-compatibility, M3 hardware/target
    constraint, M4 iteration-validity.
  * Every spec names an **outcome** (§9.6.1): `rt-check`, `avoided`,
    `unchecked`, `ct-check`, or `L`. "L" is not one of the five: it is a
    separate launch-status attribution axis, never entering an admissible
    denominator (§4). The launch/target-limit specs that carry it live in
    M3.17-M3.26 -- the former standalone `L` class was FOLDED INTO M3, because
    a launch-geometry / resource / feature-gating limit IS a target
    constraint.
  * A transform that does not change the source is a `noop` and is rejected.
  * Counts are reached by enumerating defect *magnitudes* within a spec (e.g.
    K % 16 with residues 2/4/6/8/12), never by inventing defects outside §1-§4.
  * **Budget follows the outcome** (§9.6.1): `rt-check` cells get the `-rtc`
    curve sweep; `unchecked`/`ct-check` cells get ONE injection per
    (spec x surface) cell; `avoided` and noop-by-construction specs get ZERO
    injections and are recorded as N/A.
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
# carries obligations. The target-limit specs (M3.17-M3.26, formerly class L)
# ride on conv2d because that is where the block and shared extents are
# explicit -- and conv2d is already in M3's minimal set, so they land at level 1.
MINIMAL_SET = {
    "M1": ["layer_normalization", "softmax", "relu", "transpose"],
    "M2": ["layer_normalization", "matmul", "concat", "conv2d", "relu",
           "softmax"],
    # M3 needs four categories at level 1 (method-taxonomy budget
    # `kernel_component = 4`). matmul/conv2d are the MMA/TMA carriers; the
    # resource-only families (M3-h and friends) additionally need categories
    # whose dynamic base cases expose a runtime extent on an on-chip tile, so
    # batch_norm, layer_normalization and max_pool2d ride here too. All three
    # carry a `shared` and/or `local` tile whose extent stays symbolic in the
    # dynamic build (verified with `choreo -t cc -i` + `-sa=muchk`).
    "M3": ["matmul", "conv2d", "batch_norm", "layer_normalization",
           "max_pool2d", "dma_rank5"],
    # layer_normalization and softmax are here because M1.6 (empty range /
    # zero stride) is registered as class M4, and its instances live on those
    # categories. Stage 1 of selection is a floor per (spec_id, category): a
    # declared spec that produces NO instance in its own class is indistinguish-
    # able from a forgotten spec, so the class must cover every category its
    # specs are realisable on. Before the re-homing these 7 operators sat in
    # M1's cell, where the categories were already covered -- which is exactly
    # why the gap stayed invisible until `cls` became authoritative.
    "M4": ["conv2d", "relu", "transpose", "layer_normalization", "softmax"],
}

# Level-2 widening order (specs §5), applied only after level-1 is green.
LEVEL2_SET = {
    "M1": ["max_pool2d", "conv2d", "embedding", "batch_norm"],
    # `transpose_square` is mlir-linalg's M2-e surface (family "layout (extents
    # intact)"): a square transpose keeps every extent legal under a wrong
    # permutation, so only the memory order changes. Declared here (not in
    # MINIMAL_SET) so it widens the coverage set without reclassifying any
    # existing choreo level-1 category.
    "M2": ["elemwise_add", "transpose_square"],
    "M3": ["batch_norm"],
    "M4": [],
}


# ===========================================================================
# SPEC_REGISTRY -- the spec of record, in code (specs v2.1 §1-§4, §9.6).
#
# This is GATE 1. Every mutation spec the suite claims to cover has one entry
# here, carrying the three things v2.1 added to the vocabulary:
#
#   path          rt-check | avoided | unchecked | ct-check
#                 ("L" = launch-status, attribution-only, never admissible)
#   admissible    may this spec contribute to the admissible denominator?
#                 False for noop-by-construction specs and for every L spec.
#   prohibition   if not applicable, WHY (specs §9.5.0):
#                   absent       the suite cannot express the corrupted state
#                                (no source surface); include the missing
#                                surface in `note` so the gap is actionable
#                   derived      the prohibition is not stated, only derived
#                   repaired     the model FORBIDS the state, so the compiler
#                                refuses it and there is no test to run
#                                (outcome: avoided)
#                   harness-owned the check lives in the harness, not the
#                                compiler (the E2 obligation suite)
#                   observation  attribution-only (the launch-status class):
#                                the mutant is generated but its outcome can
#                                only ever be "rejected launch", so it enters
#                                no admissible denominator
#
# TWO ORTHOGONAL AXES. `status` records whether an operator EXISTS; `admissible`
# records whether the spec may enter the denominator. They are deliberately not
# merged: M1.6/M3.6/M4.4/M3.17/M3.18/M3.20 ARE generated (their non-detection
# is the point) yet stay out of the denominator, so collapsing the two would
# either lose the controls or inflate the denominator.
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
    "M1.1": _spec("M1", "rt-check", "dropped boundary mask", status="implemented"),
    "M1.2": _spec("M1", "rt-check", "p#n off-by-one", status="implemented"),
    "M1.3": _spec("M1", "rt-check", "negative index", status="implemented"),
    "M1.4": _spec("M1", "rt-check", "transposed / non-contiguous stride",
                  status="implemented"),
    "M1.5": _spec("M1", "rt-check", "offset-view overrun", status="implemented"),
    "M1.6": _spec("M4", "rt-check", "zero-stride / empty range",
                  admissible=False, prohibition="absent",
                  status="implemented",
                  note="class is M4, not M1: the edit is on an M1 index "
                       "surface but the defect is M4's 'empty space', so "
                       "the operator counts in M4's cell (family M4-d) "
                       "while the spec_id keeps its M1 spelling. "
                       "noop by construction -- there is no prohibition to "
                       "enforce, because the range is correctly empty. "
                       "Retained for the miss audit, excluded from the "
                       "admissible denominator"),
    "M1.7": _spec("M4", "rt-check", "reversed loop bound (upper < lower) -> overrun "
                              "instead of empty",
                  note="class is M4, not M1: same re-homing as M1.6 -- the "
                       "defect is M4's 'reversed bound' (family M4-e), so "
                       "the operator must not be counted in M1's cell"),
    "M1.8": _spec("M1", "rt-check", "stride scaling (stride x 2)"),
    "M1.9": _spec("M1", "rt-check", "base offset applied WITHOUT shrinking the extent"),
    "M1.10": _spec("M1", "rt-check", "tile-boundary rounding (floor vs ceil on the "
                               "last tile)"),
    "M1.11": _spec("M1", "rt-check", "read-after-write aliasing overlap"),
    "M1.12": _spec("M1", "rt-check", "wrong loop variable for a dimension "
                               "(broadcast index reuse)"),
    "M1.13": _spec("M1", "rt-check", "symbolic-bound overrun (index beyond a bound "
                               "over a runtime parameter)"),
    "M1.14": _spec("M1", "rt-check", "chunkat tile-coordinate over/underflow while "
                               "the element index stays in bounds",
                  note="load-bearing: the only spec targeting a check the "
                       "compiler wrote and then disabled (272 chunkat "
                       "obligations, 0 enabled at -rtc=entry)"),
    "M1.15": _spec("M1", "unchecked", "dimof index >= rank, non-constant index",
                  admissible=False, prohibition="absent",
                  note="gap spec: the dimof-rank mechanism neither exists in "
                       "the source suite nor, per specs §1.1, emits any "
                       "runtime obligation (0/974). MISSING SURFACE: a case "
                       "using `dimof` with a runtime index"),
    "M1.16": _spec("M1", "unchecked", "select factor out of range (f >= count | f < 0)",
                  admissible=False, prohibition="absent",
                  note="semacheck.cpp:2032-2038, the no-static-factor path is "
                       "unassessed (defect F10) -- but no case in the suite "
                       "uses `select`, so the rule is never reached. "
                       "MISSING SURFACE: a case with a runtime select factor"),
    "M1.17": _spec("M1", "unchecked", "5th-index access on a rank-5 view",
                  admissible=False, prohibition="absent",
                  note="MISSING SURFACE: the suite's maximum rank is 4, so a "
                       "rank-5 view cannot be constructed"),
    "M1.18": _spec("M1", "rt-check", "index in range by `interval` but out of range "
                               "by `canonical` (or vice versa)",
                  admissible=False, prohibition="absent",
                  note="specs v2.1 states the intent but gives no "
                       "realization, and the two mechanisms are internal "
                       "(shapeinfer). MISSING SURFACE: a case whose index map "
                       "makes `interval` and `canonical` disagree -- must be "
                       "identified before this spec can be written"),
    "M1.19": _spec("M1", "unchecked", "index-carrier overflow (realization b, "
                               "narrow-carrier)", status="implemented",
                  note="adopted from §9.5.1; the (int) cast at "
                       "cute_codegen.cpp:1757 is also a latent shipped defect"),
    "M1.20": _spec("M1", "unchecked", "view / subspan offset, stride, rank arity "
                               "(_StaticFail_-only; symbols neither refused "
                               "nor checked)", status="implemented",
                   note="unchecked for a symbolic offset, avoided (repaired) "
                        "for a static one -- the split must be explicit or the "
                        "noop rate is an artefact"),
    "M1.21": _spec("M1", "rt-check", "tileat/at index vs the tiled extent; step/"
                               "stride tail on a non-divisible extent",
                  status="implemented",
                   note="rt-check per-dim, avoided/unchecked for the "
                        "composition"),

    # ---- M2 shape-compatibility (20 specs, specs §2) --------------------
    "M2.1": _spec("M2", "rt-check", "wrong leading extent for a secondary operand",
                  status="implemented"),
    "M2.2": _spec("M2", "rt-check", "DMA src/dst extent disagreement",
                  status="implemented"),
    "M2.3": _spec("M2", "rt-check", "binary op on mismatched shapes",
                  status="implemented"),
    "M2.4": _spec("M2", "rt-check", "wrong output leading extent",
                  status="implemented"),
    "M2.5": _spec("M2", "rt-check", "partial write (omitted tail tile) / duplicate "
                              "write (overlapping tile)", status="implemented"),
    "M2.6": _spec("M2", "rt-check", "two extents transposed"),
    "M2.7": _spec("M2", "rt-check", "reduced-rank view (a dimension dropped)"),
    "M2.8": _spec("M2", "rt-check", "broadcast extent set to 1 instead of N"),
    "M2.9": _spec("M2", "rt-check", "batch/group dimension swapped"),
    "M2.10": _spec("M2", "rt-check", "transpose permutation on a SQUARE operand "
                               "(extents equal, memory order changes)",
                   note="view/metadata family -- best-attested class in both "
                        "empirical corpora; report as its own sub-table"),
    "M2.11": _spec("M2", "rt-check", "DMA to-buffer element-count undersize on a "
                               "LogicalEqual path"),
    "M2.12": _spec("M2", "avoided", "rank mismatch through .pad (overlap still "
                               "satisfies f + pad == t)",
                   admissible=False, prohibition="repaired",
                   note="semacheck.cpp:1081-1090 is a hard Error1 producing NO "
                        "ledger row -- not applicable (specs §2.1)"),
    "M2.13": _spec("M2", "rt-check", "shape-equal / layout-unequal (every extent "
                               "agrees, the affine map does not)",
                   note="view/metadata family"),
    "M2.14": _spec("M2", "rt-check", "matmul contraction-dim (K) mismatch masked by "
                               "broadcast"),
    "M2.15": _spec("M2", "rt-check", "pad_low <-> pad_high SWAPPED (length preserved, "
                               "placement differs)", status="implemented",
                   note="semacheck.cpp:1076-1100 is a SUM -- blind to "
                        "placement; the M2 dual of M2.10"),
    "M2.16": _spec("M2", "rt-check", "span_as preserving ElementCount() with a "
                               "different rank/split", status="implemented",
                   note="semacheck.cpp:947 compares COUNT, not shape"),
    "M2.17": _spec("M2", "unchecked", "span_as / reshape on runtime-shaped data -- "
                               "check skipped entirely", status="implemented",
                   note="semacheck.cpp:946-950 guarded by !RuntimeShaped() on "
                        "both sides"),
    "M2.18": _spec("M2", "ct-check", "reshape on a non-contiguous span "
                               "(warning-only path)",
                  admissible=False, prohibition="absent",
                  note="semacheck.cpp:1195+ Warning(rop->LOC(), ...) -- the "
                       "only ct-check-from-warning spec, so this gap leaves "
                       "the warning-then-"
                       "continue path with ZERO coverage. MISSING SURFACE: a "
                       "strided view. Every reshape case applies span_as to "
                       "a contiguous parameter, and preserving "
                       "ElementCount() rules out slicing; a non-contiguous "
                       "span needs a stride the source suite never writes"),
    "M2.19": _spec("M2", "unchecked", "DMA extent mismatch where >=1 extent is "
                               "symbolic -- hard error escaped",
                  status="implemented",
                  note="semacheck.cpp:1140-1145 -- emit_error forced false"),
    "M2.21": _spec("M2", "rt-check", "MSB broadcast extent neither 1 nor equal "
                               "(rank-unequal path checks trailing dims only)",
                  note="semacheck.cpp:466-500"),

    # ---- M3 hardware-constraint (16 specs, specs §3) --------------------
    "M3.1": _spec("M3", "rt-check", "contraction extent not divisible by the "
                              "tensor-core atom (tail dropped)",
                  status="implemented"),
    "M3.2": _spec("M3", "rt-check", "descriptor dimension >= 2^24 (family A)",
                  admissible=False, prohibition="absent",
                  note="specs §3.3 requires the bound be violated by a STRIDE, "
                       "not a length (a length would allocate). The source "
                       "suite has no strided-view surface, and a descriptor "
                       "extent can only come from `.span_as`, whose "
                       "ElementCount check (M2.16, defect F6) fires first -- "
                       "so a naive realization measures M2.16, not M3.2. "
                       "MISSING SURFACE: a strided view"),
    "M3.3": _spec("M3", "rt-check", "TMA box byte-size >= 2^24 after 128-byte "
                              "ceiling (family B)",
                  admissible=False, prohibition="absent",
                  note="family B needs a 128-byte-ceiled row extent >= 2^24, "
                       "i.e. a row of >= 2^22 f32. Reaching it while "
                       "preserving ElementCount() is arithmetically "
                       "impossible (the row would exceed the whole tensor), "
                       "so every realization either allocates or trips the "
                       "count check. MISSING SURFACE: a strided view"),
    "M3.4": _spec("M3", "rt-check", "tensor footprint >= 4 GB, 5-D product "
                              "(family C)",
                  status="implemented",
                  note="gpu_adapt.hpp:351-355 -- the family-C obligation is "
                       "the rank-5 product CeilTo128Byte(bpe * dst dim0) * "
                       "dim1 * dim2 * dim3 * dim4 < 2^32. Realised on "
                       "`dma_rank5`/`r5dyn`, a rank-5 global->shared .transp "
                       "whose dim0 is symbolic, so the assessment is a "
                       "runtime_check; the mutation lifts a trailing literal "
                       "dim so the product crosses 2^32 well inside the legal "
                       "runtime range"),
    "M3.5": _spec("M3", "rt-check", "swizzle-incompatible box shape",
                  admissible=False, prohibition="absent",
                  note="MISSING SURFACE: no case in the suite writes an "
                       "explicit swizzle mode, so swizzle width <-> box "
                       "geometry is not source-expressible"),
    "M3.6": _spec("M3", "rt-check", "leading dim not aligned to descriptor "
                              "granularity ON A VECTORIZED ACCESS",
                  admissible=False, prohibition="absent",
                  status="implemented",
                  note="replaces v1 M3 s2: the v1 realization (lhs1/rhs1/w1/i1) "
                       "is a noop because the reference k_matmul is SCALAR, so "
                       "an unaligned scalar load is legal. Retained as a named "
                       "control until the vectorized realization is written."),
    "M3.7": _spec("M3", "rt-check", "TMA inner-box geometry not 128-bit aligned",
                  admissible=False, prohibition="absent",
                  note="MISSING SURFACE: box geometry is inferred by the "
                       "lowering, never written in source; the only direct "
                       "lever is an extent change, which allocates"),
    "M3.8": _spec("M3", "ct-check", "DMA rank = 6 (outside the assessed [1,5])",
                  status="implemented",
                  note="gpu_adapt.hpp:337 RankLE5 -- the rank in "
                       "dma.transp(not slice nor deslice) must be in [1,5]. "
                       "Realised on `dma_rank5`/`r5base`: a rank-5 base whose "
                       "mutation adds a 6th dim and a 6-wide permutation, so "
                       "the compiler refuses it at compile time (Error1). "
                       "ct-check is assessed, not cost-suppressible, so it "
                       "takes the one-per-cell ceiling rather than the -rtc "
                       "curve"),
    "M3.9": _spec("M3", "rt-check", "pad-field overrun (dma.pad / padding_mid beyond "
                              "the assessed range)"),
    "M3.10": _spec("M3", "rt-check", "last-dim / rank-5 mid-padding violates "
                               "padding_mid[rank-1] == 0"),
    "M3.11": _spec("M3", "rt-check", "shared operand base not 128-byte aligned on "
                               "sm_90+",
                   admissible=False, prohibition="absent",
                   note="ARCH-DEPENDENT: a noop on sm_86 (SHARED alignment "
                        "16), a mis-read on sm_90+ (128); needs per-mutant "
                        "-arch (specs §9.4 q7). MISSING SURFACE: the suite "
                        "never writes an explicit shared base offset, so the "
                        "alignment cannot be perturbed from source"),
    "M3.12": _spec("M3", "avoided", "shared tile exactly at the capacity bound "
                               "(+/-1 KiB edge)",
                   admissible=False, prohibition="repaired",
                   note="RECLASSIFIED rt-check -> avoided. memcheck.hpp:106-123 "
                        "(CheckCtMemUsage) raises Error1 when compile-time "
                        "SHARED/LOCAL usage exceeds the limit, so the model "
                        "FORBIDS the state and the compiler refuses it: there "
                        "is no test to run. A runtime channel exists only when "
                        "an extent is symbolic, which the suite's shared "
                        "tiles are not. This also explains the 28/40 M3 noops "
                        "in the committed v1 baseline -- v1 s3 was exactly "
                        "this family (specs §3.0)"),
    "M3.13": _spec("M3", "rt-check", "swizzle width <-> box inner dim <-> shared "
                               "alignment, narrowed",
                   admissible=False, prohibition="absent",
                   note="the injected state must SURVIVE the compiler's own "
                        "repair at cute_codegen.cpp:10271 -- only the "
                        "sub-case the repair misses is injectable (R3), and "
                        "no case in the suite writes an explicit swizzle. "
                        "MISSING SURFACE: a swizzled shared descriptor"),
    "M3.14": _spec("M3", "unchecked", "linear .copy with a dimension >= 2^24 -- the "
                               "check is absent", status="implemented",
                   note="gpu_adapt.hpp:320 `// linear copy` ... `// omitted`; "
                        "the other six cells of the DMA matrix call "
                        "CheckDimSize (defect F1)"),
    "M3.15": _spec("M3", "unchecked", ".pad with a dimension >= 2^24 -- the pad path "
                               "never calls CheckDimSize", status="implemented",
                   note="gpu_adapt.hpp:360-450 (defect F2)"),
    "M3.16": _spec("M3", "unchecked", "TMA box inner alignment with a SYMBOLIC leading "
                               "dim -- no assessment", status="implemented",
                   note="gpu_adapt.hpp:640 `// TODO: emit runtime assessment` "
                        "(defect F3)"),

    # ---- M4 iteration-validity (5 specs, specs §9.1) --------------------
    "M4.1": _spec("M4", "rt-check", "with-in mdspan dim mutated to 0 -- control",
                  note="LoopBound, forced to ENTRY cost, enabled 11/11"),
    "M4.2": _spec("M4", "rt-check", "parallelby bound mutated to 0 (a legal but "
                           "empty iteration space)", status="implemented"),
    "M4.6": _spec("M4", "rt-check", "parallelby bound mutated to negative -- "
                           "invalid, so unlike an empty space it must be "
                           "rejected", status="implemented",
                  note="split out of M4.2 so one operator is not evidence for "
                       "two families: M4-a (zero bound) and M4-b (negative "
                       "bound) have different oracle expectations, and while "
                       "they shared a spec_id M4.2 was the sole realisation "
                       "of M4-b and one of M4-a's two (defect D1)"),
    "M4.3": _spec("M4", "rt-check", "parallelby bound symbolic and zero only at "
                              "runtime"),
    "M4.4": _spec("M4", "rt-check", "bound > 0 but the iteration space is empty "
                              "(zero-trip loop)", admissible=False,
                  prohibition="absent",
                  note="deliberate noop control -- there is nothing to "
                       "forbid: a zero-trip loop is legal and its body never "
                       "executes. If it is ever counted as admissible the "
                       "oracle has regressed"),
    "M4.5": _spec("M4", "rt-check", "stride/step = 0 in an iteration"),
    "M4.7": _spec("M4", "rt-check", "padded extent goes NEGATIVE -- the "
                          "negative-padding corner the GSC study names",
                  status="implemented",
                  note="re-realises M2.12's negative-padding trigger on "
                       "rt-check (method-taxonomy.json `reassigned.M2.12`): the "
                       "rank repair through `.pad` was avoided/`repaired` and so never "
                       "generated. The edit is on the padded-extent "
                       "derivation, not on a loop literal, which is what keeps "
                       "it distinct from the empty-range control M4-d"),
    "M4.8": _spec("M4", "rt-check", "padded extent goes EMPTY -- the zero-length "
                          "half of the GSC corner-case trigger",
                  status="implemented",
                  note="the second rt-check realisation of M4-g, so the family is "
                       "not evidence from a single operator (defect D1). An "
                       "empty padded extent is a legal zero-trip loop whose "
                       "body never runs, so the expected verdict is a miss, "
                       "not a refusal"),

    # ---- M3.17-M3.26 target limits (were class L) ------------------------
    # FOLDED INTO M3: the standalone `L launch-status` class was merged here,
    # because a launch-geometry (K3), per-SM-resource (K4) or feature-gating
    # (K5) limit IS a target hardware/software constraint. The "L" *path* is
    # retained: these never enter an admissible denominator. Their value is
    # that they grow the never-attribution table, showing that a miss is
    # usually a REJECTED LAUNCH rather than a wrong answer. The class axis
    # stays M1-M4 (there is NO M5): the launch/rejection outcome is reported by
    # the E5 dynamic-oracle experiment, not as a mutation class.
    "M3.17": _spec("M3", "L", "shared-memory tile exceeds the device limit",
                admissible=False, prohibition="observation",
                status="implemented",
                note="migrated out of v1 M3 s3 (shared32/64/128, "
                     "shared512/1024); observed as launch rejected. KEPT "
                     "VERBATIM: the committed E1 baseline defines this "
                     "realization, and a reclassification would invalidate "
                     "the frozen S1/S2. The memcheck Error1 finding under "
                     "M3.12 argues it should be re-examined in v3"),
    "M3.18": _spec("M3", "L", "thread-block / cluster extent exceeds the device "
                          "limit", admissible=False,
                prohibition="observation", status="implemented",
                note="hosts v1 M3 s3's per-thread accumulator oversize "
                     "(local8/local16): same observation channel (launch "
                     "rejected), the §4 text names block/cluster extent. "
                     "KEPT VERBATIM for the same reason as L1"),
    "M3.19": _spec("M3", "avoided", "__launch_bounds__ understated vs actual block "
                "size", admissible=False, prohibition="repaired",
                note="was L3, upgraded absent -> repaired on direct evidence: "
                     "`choreo -gs -t cute -arch=sm_86` on "
                     "tests/gpu/codegen/cute/launch_bounds.co reports `error: "
                     "[[launch_bounds]] maxThreadsPerBlock (8) is less than "
                     "the computed thread count (64).` The model DERIVES the "
                     "required extent and refuses any understatement, so there "
                     "is no test to run (avoided). The suite still never writes the "
                     "attribute, but that is no longer the operative reason"),
    "M3.20": _spec("M3", "L", "block extent not a multiple of 32 / of 128",
                admissible=False, prohibition="observation",
                note="the one L spec with an operator: the block extent is "
                     "source-visible (`parallel p by N`), so this class is "
                     "exercised end to end"),
    "M3.21": _spec("M3", "avoided", "shared tile exceeds per-SM capacity",
                admissible=False, prohibition="repaired",
                note="RECLASSIFIED: the v2.1 L-class membership assumed the "
                     "launch-rejected channel, but the check the compiler "
                     "actually performs is memcheck.hpp:106-123 CheckCtMemUsage "
                     "-> Error1, a hard compile-time refusal, so the model forbids it "
                     "(avoided). Same "
                     "mechanism as M3.12, one step past the limit"),
    "M3.22": _spec("M3", "avoided", "WGMMA used below sm_90", admissible=False,
                prohibition="repaired",
                note="was L6, upgraded absent -> repaired on direct evidence: "
                     "`choreo -gs -t cute -arch=sm_86` on "
                     "tests/gpu/end2end/wgmma_ss.co reports `error: group-4 "
                     "level is not supported by the target architecture: "
                     "sm_86.` (x14); the same file compiles clean under "
                     "`-arch=sm_90a`. This is the K5 feature-gating discharge: "
                     "the compiler refuses the feature below sm_90, so there "
                     "is no test to run (avoided)"),
    "M3.23": _spec("M3", "L", "shared operand used outside WGMMA", admissible=False,
                prohibition="absent",
                note="MISSING SURFACE: no WGMMA in the suite, so 'outside "
                     "WGMMA' has no referent"),
    "M3.24": _spec("M3", "L", "unsupported MMA configuration", admissible=False,
                prohibition="absent",
                note="MISSING SURFACE: no explicit MMA in the suite -- the "
                     "reference k_matmul is a scalar helper, not a tensor-"
                     "core intrinsic"),
    "M3.25": _spec("M3", "L", "mma.scale operand is not an accumulator",
                admissible=False, prohibition="absent",
                note="MISSING SURFACE: no mma.scale in the DSL surface the "
                     "suite uses"),
    "M3.26": _spec("M3", "L", "cluster extent > 8 (non-portable)", admissible=False,
                 prohibition="absent",
                 note="MISSING SURFACE: no case in the suite declares a "
                      "cluster"),

    # ---- M3.27/M3.28 on-chip capacity on a SYMBOLIC extent (rt-check) ----
    # The rt-check realisations of family M3-h. M3.12 is the declared one and is
    # avoided/repaired: CheckCtMemUsage refuses a tile whose byte size it can fold,
    # and M3.12's own note draws the consequence -- "a runtime channel exists
    # only when an extent is symbolic". These two supply that extent, so the
    # tile is not refused at compile time and the family is no longer
    # unrealisable as declared. Distinct from M3.17/M3.18 (path L, prohibition
    # `observation`): these are admissible and can witness a miss.
    "M3.27": _spec("M3", "rt-check", "shared tile exceeds the device budget, but only "
                          "for a runtime-shaped (symbolic) extent",
               status="implemented",
               note="memcheck.hpp:267-287 (CheckCtMemUsage) compares "
                    "COMPILE-TIME byte totals against the limit and raises "
                    "Error1 (that is M3.12). A runtime-shaped extent is not a "
                    "constant, so the total it contributes is not a constant "
                    "either and the check cannot fire: the oversized tile is "
                    "launched, and only a runtime channel can judge it"),
    "M3.28": _spec("M3", "rt-check", "per-thread local tile exceeds the per-thread "
                          "budget, but only for a runtime-shaped (symbolic) "
                          "extent",
               status="implemented",
               note="the LOCAL half of M3-h, same mechanism as M3.27. The "
                    "static check has two local bands (memcheck.hpp:288-306, "
                    "warn-only between the best-practice budget and the hard "
                    "cap), so an over-budget local tile is exactly where a "
                    "compiler that only checks SHARED stays silent"),
}

# The v1 spec integers still carried by every operator (specs v1 §1-§3) map 1:1
# onto v2.1 ids for M1 and M2, and onto a REVISED target for M3:
#   M3 s1 -> M3.1   (retained verbatim)
#   M3 s2 -> M3.6   (noop; the vectorized realization is M3.6)
#   M3 s3 -> M3.17/M3.18 (resource-exhaustion on the launch-status path; the
#                former L1/L2 -- shared -> M3.17, per-thread local -> M3.18)
# Prefer an explicit `spec_id=` on new operators; this table exists so the v1
# corpus stays attributable without rewriting 117 call sites.
V1_SPEC_ID = {
    ("M1", 1): "M1.1", ("M1", 2): "M1.2", ("M1", 3): "M1.3",
    ("M1", 4): "M1.4", ("M1", 5): "M1.5",
    # v1 M1 s6/s7 are re-homed to class M4. The edit is on an M1 index
    # surface, but the defect is M4's bound semantics ("empty space",
    # "reversed bound"), so their operators must be counted in M4's cell --
    # cf. `families.M4-d` / `families.M4-e` and `reassigned` in
    # method-taxonomy.json. This map is keyed on the cls ARGUMENT, which is
    # exactly why the key moves to M4 while the spec_id keeps its M1
    # spelling. Leaving the key at ("M1", 6) would make the 13 operators
    # fall through to the identity fallback "M4.6", which is now the
    # negative-bound spec: one operator would become evidence for the wrong
    # family.
    ("M4", 6): "M1.6", ("M4", 7): "M1.7",
    ("M2", 1): "M2.1", ("M2", 2): "M2.2", ("M2", 3): "M2.3",
    ("M2", 4): "M2.4", ("M2", 5): "M2.5",
    ("M3", 1): "M3.1", ("M3", 2): "M3.6", ("M3", 3): "M3.17",
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
      path_class    rt-check | avoided | unchecked | ct-check | L
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
        # v1 M3 s3 is resource-exhaustion on the launch-status path. The
        # observation channel splits it: shared -> M3.17, per-thread -> M3.18.
        if sid == "M3.17" and "local" in mid:
            sid = "M3.18"
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

        Only `avoided` (the retired P2). Admissibility is a separate,
        reporting-only concept: an L operator IS generated (it grows the
        never-attribution table) and is merely excluded from the admissible
        denominator.
        """
        return self.path_class == "avoided"

    @property
    def needs_rtc_curve(self):
        """`rt-check` is the only outcome a threshold can reach
        (specs §9.6.1)."""
        return self.path_class == "rt-check"

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
    _m("M1.s6.ln1.empty", "M4", 6, "stride", "layer_normalization",
       "1_bert_32x512x768_768_768",
       "empty iteration range (iteration-validity residue)",
       ("foreach {j, k} in [J, K]", "foreach {j, k} in [J, 0]")),
    _m("M1.s6.ln1.zerostride", "M4", 6, "stride", "layer_normalization",
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
    _m("M1.s6.ln3.empty", "M4", 6, "stride", "layer_normalization",
       "3_attention_32xNx512x64_64_64",
       "empty reduction range over the dynamic extent",
       ("foreach l in [L]", "foreach l in [0]")),
    _m("M1.s5.ln3.out", "M1", 5, "oob", "layer_normalization",
       "3_attention_32xNx512x64_64_64",
       "offset view on the output chunk index",
       ("out.at(p#n, j, k, l)", "out.at(p#n + 1, j, k, l)")),
    _m("M1.s6.ln3.overrange", "M4", 6, "stride", "layer_normalization",
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
    _m("M1.s6.sm1.empty", "M4", 6, "stride", "softmax",
       "1_bert_32x512x768_32x512x768",
       "empty reduction range",
       ("foreach {k} in [l1_input.span(2)]", "foreach {k} in [0]")),
    _m("M1.s5.sm1.store", "M1", 5, "oob", "softmax",
       "1_bert_32x512x768_32x512x768",
       "offset write-back chunk overruns the output tensor",
       ("dma.copy l1_out => output.chunkat(i#p, q, _)",
        "dma.copy l1_out => output.chunkat(i#p + 1, q, _)")),
    _m("M1.s6.sm1.zerostride", "M4", 6, "stride", "softmax",
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
    _m("M1.s6.sm11.empty", "M4", 6, "stride", "softmax",
       "11_dynamic_32xSx768_32xSx768",
       "empty reduction range",
       ("foreach {k} in [l1_input.span(2)]", "foreach {k} in [0]")),
    _m("M1.s5.sm11.store", "M1", 5, "oob", "softmax",
       "11_dynamic_32xSx768_32xSx768",
       "offset write-back chunk overruns the output tensor",
       ("dma.copy l1_out => output.chunkat(i#p, q, _)",
        "dma.copy l1_out => output.chunkat(i#p + 1, q, _)")),
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
    _m("M1.s6.rl1.empty", "M4", 6, "stride", "relu",
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
    _m("M1.s6.rl11.empty", "M4", 6, "stride", "relu",
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
    _m("M1.s6.tp1.empty", "M4", 6, "stride", "transpose",
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
    _m("M1.s6.tp11.empty", "M4", 6, "stride", "transpose",
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


    # --- additional realisation cells (auto-derived: an existing
    # spec edit applied to a further base kernel where its anchor occurs,
    # raising the family to the depth its kernel spread allows) ---
    _m("M1.8.ln1.alt", 'M1', 8, 'stride', 'layer_normalization',
       '11_dynamic_32xSx768_768_768',
       'stride scaling: the row index advances by 2 within the chunk',
       ('lhs.at(p#n, j, k)', 'lhs.at(p#n, j * 2, k)', None)),
    _m("M1.12.ln1.alt", 'M1', 12, 'stride', 'layer_normalization',
       '11_dynamic_32xSx768_768_768',
       'wrong loop variable for a dimension on the primary operand',
       ('lhs.at(p#n, j, k)', 'lhs.at(p#n, j, j)', None)),
    _m("M1.12.sm1.alt", 'M1', 12, 'stride', 'softmax',
       '11_dynamic_32xSx768_32xSx768',
       'wrong loop variable for a dimension: the reduction index is reused for the row dimension (broadcast index reuse)',
       ('l1_input.data.at(0, j, k)', 'l1_input.data.at(0, k, k)', None)),
    _m("M1.21.ln1.alt", 'M1', 21, 'oob', 'layer_normalization',
       '11_dynamic_32xSx768_768_768',
       'tile index one past the tiled extent on the primary operand',
       ('lhs.at(p#n, j, k)', 'lhs.at(p#n, j, k + K)', None)),
    _m("M1.11.rl1.alt", 'M1', 11, 'stride', 'relu',
       '11_dynamic_32xSx768_32xSx768',
       "read-after-write aliasing: the shared tile is shrunk below the `parallel q by 64` fan-out, so threads overwrite each other's region",
       ('shared f32 [1, 1, 64, 1] inp_s, out_s;', 'shared f32 [1, 1, 32, 1] inp_s, out_s;', None)),
    _m("M1.11.tp1.alt", 'M1', 11, 'stride', 'transpose',
       '11_dynamic_32xSx768_32x768xS',
       'read-after-write aliasing on the transpose source tile',
       ('shared f32 [1, 1, 64] is;', 'shared f32 [1, 1, 32] is;', None)),
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
       ("l1_a = dma.copy lhs.chunkat(p#q, m_tile, k_tile) => shared;",
        "l1_a = dma.copy rhs.chunkat(p#q, m_tile, k_tile) => shared;")),
    _m("M2.s2.mm11.tiles", "M2", 2, "dim-mismatch", "matmul",
       "11_dynamic_32xSx768_768x768_32xSx768",
       "tile count disagrees with the tiled extent on the n dimension",
       ("with index = {m_tile, n_tile, k_tile} in [32, 32, 32] {",
        "with index = {m_tile, n_tile, k_tile} in [32, 33, 32] {")),
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

    # ------------------------------------------------------------------
    # Second realisation cells for the M2 families whose ceiling was not
    # reached because a family's depth is bounded by the number of KERNELS
    # its specs' edit anchors exist on (select() caps 2 per category):
    #   family depth <= 2 x (#categories the family's anchors exist on)
    # Each operator below is an EXISTING spec's edit list applied to a
    # further base kernel where its anchor is present -- no new semantics.
    # ------------------------------------------------------------------
    _m("M2.11.rl1.alt", "M2", 11, 'wrong-shape', 'relu',
       '11_dynamic_32xSx768_32xSx768',
       'DMA destination buffer undersized by one element on an otherwise LogicalEqual path on relu (second cell: the same defect on another kernel of the family)',
       ('shared f32 [1, 1, 64, 1] inp_s, out_s;', 'shared f32 [1, 1, 63, 1] inp_s, out_s;')),
    _m("M2.9.ln1.alt", "M2", 9, 'dim-mismatch', 'layer_normalization',
       '6_dynamic_128xCx112x112_112x112_112x112',
       'batch/group dimension swapped on the primary operand on layer_normalization (second cell: the same defect on another kernel of the family)',
       ('f32 [I, N0, K, L] lhs', 'f32 [N0, I, K, L] lhs')),
    _m("M2.6.mm1.alt", "M2", 6, 'dim-mismatch', 'matmul',
       '11_dynamic_32xSx768_768x768_32xSx768',
       'two leading extents transposed in the output declaration on matmul (second cell: the same defect on another kernel of the family)',
       ('f32 [lhs.span(0), lhs.span(1), rhs.span(1)] output;', 'f32 [lhs.span(1), lhs.span(0), rhs.span(1)] output;')),
    _m("M2.7.cc1.alt", "M2", 7, 'dim-mismatch', 'concat',
       '15_gpt_16x512x1536_16x512x1536_16x512x3072',
       'reduced-rank view on the second concat operand on concat (second cell: the same defect on another kernel of the family)',
       ('f32 [I, J, K, L] b)', 'f32 [I, J, K] b)')),
    _m("M2.16.cv1.alt", "M2", 16, 'dim-mismatch', 'conv2d',
       '20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1',
       'span_as split exchanged: ElementCount is preserved, so the count-only comparison at semacheck.cpp:947 passes on conv2d (second cell: the same defect on another kernel of the family)',
       ('dma.copy w.span_as(Cout, K).chunkat(_, kt) => B_tile;', 'dma.copy w.span_as(K, Cout).chunkat(_, kt) => B_tile;')),
    _m("M2.7.mm1.alt", "M2", 7, 'dim-mismatch', 'matmul',
       '11_dynamic_32xSx768_768x768_32xSx768',
       'reduced-rank view: a dimension is dropped from the output declaration on matmul (second cell: the same defect on another kernel of the family)',
       ('f32 [lhs.span(0), lhs.span(1), rhs.span(1)] output;', 'f32 [lhs.span(0), rhs.span(1)] output;')),
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
             ("with index = {m_tile, n_tile, k_tile} in [32, 32, 32] {",
              "with index = {m_tile, n_tile, k_tile} in [32, 32, 31] {")))
M3.append(_m("M3.s2.mm11.rhs1", "M3", 2, "stride", "matmul",
             "11_dynamic_32xSx768_768x768_32xSx768",
             "misaligned tensor-core base: rhs DMA chunk offset by 1",
             ("l1_b = dma.transp<1,0> rhs.chunkat(k_tile, n_tile) => shared;",
              "l1_b = dma.transp<1,0> rhs.chunkat(k_tile, n_tile + 1) => shared;")))
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
                 ("f32 [32, 128, 1, 1] w", f"f32 [32, {_cin}, 1, 1] w")))

# `ALL` is assembled once every class list exists -- see the grouping pass
# after `M4`. It is deliberately NOT built here, because the class a mutant
# counts in is `Mut.cls`, not the list it happens to be declared in, and the
# two disagree for the 13 re-homed M1.6/M1.7 operators.


def transforms_for(cls):
    """The operators that count in `cls`'s cell. `ALL` is built after M4."""
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
# matched, so GATE 2 is a check rather than a hope. Budget follows the outcome
# (§9.6.1): rt-check specs get several operators per category so the -rtc curve
# has something to sweep; unchecked/ct-check specs get ONE per
# (spec x category) cell, because no obligation exists at any threshold.
# ===========================================================================

# ---- M1.7 reversed loop bound -> overrun, not empty -----------------------
M1 += [
    _m("M1.7.rl1.revbound", "M4", 7, "stride", "relu",
       "1_bert_32x512x768_32x512x768",
       "reversed loop extent: the outer takes the innermost bound, so the "
       "tile index overruns instead of the tail being correctly empty",
       ("foreach {i, j, k} in [I / #p, J, 12]",
        "foreach {i, j, k} in [12, J, I / #p]")),
    _m("M1.7.tp11.revbound", "M4", 7, "stride", "transpose",
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
       ("dma.copy l1_out => output.chunkat(i#p, q, _)",
        "dma.copy l1_out => output.chunkat(i#p + 1, q, _)")),
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
# at every -rtc. unchecked -> one injection per cell.
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

# ---- M2.12 rank mismatch through .pad -- N/A (avoided, repaired) ----------
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
       ("l1_b = dma.transp<1,0> rhs.chunkat(k_tile, n_tile) => shared;",
        "l1_b = dma.transp<1,0> rhs.chunkat(n_tile, k_tile) => shared;")),
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
# symbolic operand escapes the size check entirely. unchecked -> one per cell.
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

# ---- M2.18 reshape on a non-contiguous span -- ct-check (warning) ---------
# semacheck.cpp:1195+ emits `Warning(rop->LOC(), ...)` and continues. ct-check
# specs need one injection per cell, and the finding is the warning, not the miss.

# ---- M2.19 DMA extent mismatch with a symbolic extent ---------------------
# semacheck.cpp:1140-1145 forces emit_error false when either extent is
# symbolic: the hard error is escaped. unchecked.
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
# the DMA matrix call CheckDimSize; this one does not (defect F1). unchecked.
#
# M3-b's ceiling is 3. `unchecked` gives ONE instance per (spec x category)
# cell (`ceiling()` returns N_CELLS, not N_REALISATIONS, when
# `needs_rtc_curve` is false), so the family's whole budget is:
#
#   M3.14 x {matmul, conv2d} + M3.15 x {conv2d} = 3, and cat_used["conv2d"]
#   is then 2 == N_REALISATIONS, which closes the family.
#
# The ceiling is 3, not 8, and that is a fact about the suite rather than
# about this file: M3.14's plain-tile anchor appears in matmul and nowhere
# else, its im2col anchor in conv2d and nowhere else, and `dma.pad` (M3.15)
# appears in conv2d and nowhere else, so `kernel_component = 4` cannot be
# reached in M3-b by any amount of operator writing. Widening
# MINIMAL_SET["M3"] to its current six kernels does not change this -- the
# binding constraint here is anchor supply, not the coverage set. The two
# realisations below are the same missing CheckDimSize reached through the
# im2col path and the plain-tile path respectively -- one defect, two lowering
# entry points.
M3 += [
    _m("M3.14.mm1.linearcopy", "M3", 14, "dim-mismatch", "matmul",
       "1_bert_32x512x768_768x768_32x512x768",
       "linear .copy with a dimension >= 2^24 reached by STRIDE, on the one "
       "DMA-matrix cell that carries no CheckDimSize call",
       ("l1_a = dma.copy lhs.chunkat(p#q, m_tile, k_tile) => local;",
        "l1_a = dma.copy lhs.chunkat(p#q, m_tile, 16777216) => local;")),
    _m("M3.14.cv1.linearcopy", "M3", 14, "dim-mismatch", "conv2d",
       _C1,
       "same absent CheckDimSize cell reached through the im2col linear copy: "
       "the K-tile index is a 2^24 stride into a tensor whose K extent is 512",
       ("l1_A = dma.copy i.chunkat(p#n, _, _, _).span_as(K, Ho, Wo)"
        ".chunkat(kt, q, _).span_as(8, M/#q) => local;",
        "l1_A = dma.copy i.chunkat(p#n, _, _, _).span_as(K, Ho, Wo)"
        ".chunkat(16777216, q, _).span_as(8, M/#q) => local;")),
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
# (defect F3). unchecked.
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
# assessed path (rt-check), unlike M3.14/M3.15 which sit on the unchecked 2^24
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

# ---- M3.27/M3.28 on-chip capacity with a SYMBOLIC extent -----------------
# M3.12's scope note is exact: CheckCtMemUsage only refuses a tile whose byte
# size it can fold, so "a runtime channel exists only when an extent is
# symbolic". These two operators supply that extent by multiplying a shared and
# a local tile by a genuinely runtime span. The tile is then oversized at run
# time while the compile-time total the checker folds stays small, so the
# capacity check cannot fire -- the gap M3-h names.
#
# The multiplier MUST be a runtime parameter, not any `.span()`. In this case
# `N = i.span(0)` is the literal batch 32, so `8 * N` constant-folds to 256 and
# the tile becomes a 1 MB compile-time constant that CheckCtMemUsage rejects
# (verified: `choreo -t cute -sa=muchk` errors "shared memory OUT OF BOUND!",
# so an `N`-based operator is compile-DETECTED, not the rt-check miss it claims).
# `H = i.span(2)` is `attn_h`, a kernel parameter in the dynamic build, so
# `8 * H` stays symbolic (`[1024, (::CONV2D::attn_h * 8)]`) and the check is
# silent.
M3 += [
    _m("M3.27.cv1.symshared", "M3", 27, "dim-mismatch", "conv2d", _C1,
       "shared tile extent made runtime-shaped (Cout x 8*H, H = attn_h): the "
       "byte size is not a compile-time constant, so the capacity check stays "
       "silent and the tile is launched far past the device budget",
       ("shared f32 [Cout, 8] B_tile;",
        "shared f32 [Cout, 8 * H] B_tile;"), spec_id="M3.27"),
    _m("M3.28.cv1.symlocal", "M3", 28, "dim-mismatch", "conv2d", _C1,
       "per-thread local tile extent made runtime-shaped (M/#q x Cout*H): the "
       "per-thread budget is exceeded in a size the static check cannot fold",
       ("local f32 [M/#q, Cout] l1_Y{0.0f};",
        "local f32 [M/#q, Cout * H] l1_Y{0.0f};"), spec_id="M3.28"),
]

# M3.27/M3.28 on the other three dynamic categories. Each base case exposes a
# different runtime symbol, so the multiplier is per case:
#   batch_norm          H/W  (h_value/w_value, kernel parameters)
#   layer_normalization N1/N2
#   matmul              N
#   max_pool2d          channels/height/width
# Every edit was verified to stay symbolic under `choreo -t cc -i` and to leave
# CheckCtMemUsage (target `muchk`) silent -- the rt-check capacity miss M3-h names.
_BN10 = "10_dynamic_16x512xHxW_512_512_16x512xHxW"
_LN10 = "10_dynamic_16x512xHxW_HxW_HxW"
_MM10 = "10_dynamic_128x1280_1280xN_128xN"
_MP10 = "10_dynamic_32xCxHxW_32xCxHd5xWd5"
M3 += [
    _m("M3.27.bn10.symshared", "M3", 27, "dim-mismatch", "batch_norm", _BN10,
       "shared reduction tile extent made runtime-shaped (H = h_value): the "
       "byte size is not a compile-time constant, so the capacity check is "
       "silent while the tile is launched past the device budget",
       ("shared f32 [1] s_mean;",
        "shared f32 [H] s_mean;"), spec_id="M3.27"),
    _m("M3.28.bn10.symlocal", "M3", 28, "dim-mismatch", "batch_norm", _BN10,
       "per-thread tile extent made runtime-shaped (H): the per-thread budget "
       "is exceeded in a size the static check cannot fold",
       ("local f32 [1] df;",
        "local f32 [H] df;"), spec_id="M3.28"),
    _m("M3.27.ln10.symshared", "M3", 27, "dim-mismatch",
       "layer_normalization", _LN10,
       "shared reduction tile extent made runtime-shaped (N1): the byte size is "
       "not a compile-time constant, so the capacity check is silent",
       ("shared f32 [1] s_mean;",
        "shared f32 [N1] s_mean;"), spec_id="M3.27"),
    _m("M3.28.ln10.symlocal", "M3", 28, "dim-mismatch",
       "layer_normalization", _LN10,
       "per-thread tile extent made runtime-shaped (N1): the per-thread budget "
       "is exceeded in a size the static check cannot fold",
       ("local f32 [1] df;",
        "local f32 [N1] df;"), spec_id="M3.28"),
    _m("M3.28.mm10.symlocal", "M3", 28, "dim-mismatch", "matmul", _MM10,
       "per-thread accumulator extent made runtime-shaped (N): the per-thread "
       "budget is exceeded in a size the static check cannot fold",
       ("rhs.span(1)/#n_tile/#q] l1_out",
        "rhs.span(1)/#n_tile/#q * rhs.span(1)] l1_out"), spec_id="M3.28"),
    _m("M3.28.mp10.symlocal", "M3", 28, "dim-mismatch", "max_pool2d", _MP10,
       "per-thread output tile extent made runtime-shaped (height): the "
       "per-thread budget is exceeded in a size the static check cannot fold",
       ("l1_input.span(3)/5] l1_out;",
        "l1_input.span(3)/5 * l1_input.span(2)] l1_out;"), spec_id="M3.28"),
]

# ---- M3.4 / M3.8 -- the rank-5 DMA surface -------------------------------
# The suite's maximum rank was 4, so family C (a rank-5 footprint product) and
# the rank-6 out-of-range side of RankLE5 had no source case at all. `dma_rank5`
# is a mutation-only category carrying the two rank-5 hosts:
#   r5dyn   global->shared .transp with a symbolic dim0 -- family C is assessed
#           at run time, so lifting a trailing literal dim moves the violation
#           into the legal runtime range (rt-check).
#   r5base  a well-formed rank-5 .transp; adding a 6th dim and a 6-wide
#           permutation takes the rank outside [1,5] (RankLE5, ct-check).
# Both base cases compile rc=0 under the pinned flags, and both mutated forms
# were verified to reach the named assessment before the edits were written.
M3 += [
    _m("M3.4.dma5dyn.footprint", "M3", 4, "stride", "dma_rank5", "r5dyn",
       "rank-5 footprint product lifted past 4 GB by a trailing literal dim: "
       "the family-C obligation is assessed at run time (dim0 symbolic) and the "
       "mutant violates it well inside the legal runtime range",
       ("f32 [N,2,1,1,1] input", "f32 [N,4096,1,1,1] input", 1),
       ("f32 [N,1,1,1,2]", "f32 [N,1,1,1,4096]", 2)),
    _m("M3.8.dma5.rank6", "M3", 8, "dim-mismatch", "dma_rank5", "r5base",
       "DMA rank raised to 6: a 6th dim and a 6-wide permutation take the "
       "descriptor outside the assessed [1,5], so the compiler refuses it at "
       "compile time (RankLE5)",
       ("f32 [2,4,8,16,32] input", "f32 [2,4,8,16,32,2] input", 1),
       ("f32 [4,8,16,32,2] output", "f32 [4,8,16,32,2,2] output", 1),
       ("dma.transp<1,2,3,4,0>", "dma.transp<1,2,3,4,5,0>", 1)),
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


    # --- additional realisation cells (auto-derived: an existing
    # spec edit applied to a further base kernel where its anchor occurs,
    # raising the family to the depth its kernel spread allows) ---
    _m("M4.6.rl1.alt", 'M4', 2, 'stride', 'relu',
       '11_dynamic_32xSx768_32xSx768',
       'parallelby bound mutated to negative on the relu inner loop',
       ('parallel q by 64', 'parallel q by -64', None), spec_id="M4.6"),
    _m("M4.6.tp1.alt", 'M4', 2, 'stride', 'transpose',
       '11_dynamic_32xSx768_32x768xS',
       'parallelby bound mutated to negative on the transpose inner loop',
       ('parallel q by 64', 'parallel q by -64', None), spec_id="M4.6"),
    _m("M4.3.rl1.alt", 'M4', 3, 'stride', 'relu',
       '18_resnet_64x256x56x56_64x256x56x56',
       'symbolic leading bound cancelled on a rank-4 relu',
       ('foreach {i, j, k, l} in [I / #p, J, K, 7]', 'foreach {i, j, k, l} in [I - I, J, K, 7]', None)),
    _m("M4.5.cv1.alt", 'M4', 5, 'stride', 'conv2d',
       '18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D',
       'zero stride in the index map: every iteration reads tile 0, so the write-back overwrites a single tile N times',
       ('dma.copy l1_Y => Y.chunkat(q#_q, qq);', 'dma.copy l1_Y => Y.chunkat(q#_q, qq * 0);', None)),
    _m("M4.5.rl1.alt", 'M4', 5, 'stride', 'relu',
       '17_mobilenet_128x96x112x112_128x96x112x112',
       'element index of the relu write-back multiplied by 0, so a single output element is overwritten N times',
       ('out.chunkat(p#i, j, k, l);', 'out.chunkat(p#i, j, k, l * 0);', None)),
]

M4 += [
    _m("M4.2.cv10.zero", "M4", 2, "stride", "conv2d",
       "10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D",
       "parallelby bound mutated to 0: a legal empty iteration space that "
       "still carries the whole body's obligations",
       ("parallel q by 8  {", "parallel q by 0  {")),
    _m("M4.6.cv1.negative", "M4", 2, "stride", "conv2d",
       _C1, "parallelby bound mutated to negative -- an invalid bound that "
            "must be rejected, unlike the legal-empty M4.2 zero cases",
       ("parallel p by 2  {", "parallel p by -2  {"), spec_id="M4.6"),
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

# ---- M4-g degenerate pad (specs §4; M2.12's trigger re-realised on rt-check) ----
# The im2col loop runs over the padded extent. Mutating the derivation of that
# extent -- not the loop literal, which is M4-a/M4.1's surface -- makes the loop
# bound degenerate at its source, in the padding arithmetic. This is the
# negative-padding corner case of the GSC study: M2.12 routed it to a avoided rank
# repair and so never generated it, which left the family with no realisation
# at all.
M4 += [
    _m("M4.7.cv10.negpad", "M4", 7, "stride", "conv2d",
       "10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D",
       "padded extent computed as 2*padding - H - 1: negative for the case's "
       "own padding (=1) and H (=16), so the im2col loop over the padded "
       "extent is invalid rather than merely empty -- the negative-padding "
       "trigger the GSC study names",
       ("Hpad = H + 2 * padding;", "Hpad = 2 * padding - H - 1;"),
       spec_id="M4.7"),
    _m("M4.8.cv14.emptypad", "M4", 8, "stride", "conv2d",
       "14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D",
       "padded extent vanishes exactly (W - W): the zero-length half of the "
       "GSC corner-case trigger, a legal zero-trip loop whose body never runs",
       ("Wpad = W + 2 * padding;", "Wpad = W - W;"),
       spec_id="M4.8"),
]

# ---- M4 depth -----------------------------------------------------------
# The four iteration-validity states, realized across the rest of M4's minimal
# set (conv2d / relu / transpose) so that "M4 is flat" is a measured property
# of the class rather than an artifact of a few hand-picked edits.
#
# M4 is the COST-FILTER CONTROL. Every mutant below leaves a legal-or-refused
# iteration space whose body never executes (M4.1/M4.2/M4.3) or whose writes
# all collide on one tile (M4.5). The expected verdict is therefore ENTRY cost
# and "not attributed": if any of these shows a curve, the cost filter is
# broken, which is exactly what the control is for. None of them can be
# repaired by a value check, so none is a coverage claim.
M4 += [
    # M4.1 -- an extent in the loop nest mutated to 0.
    _m("M4.1.cv14.zeroextent", "M4", 1, "stride", "conv2d",
       "14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D",
       "with-in mdspan extent mutated to 0 on the padded-input view",
       ("with {i, j} in [Hpad, Wpad]", "with {i, j} in [0, Wpad]")),
    _m("M4.1.cv16.zeroextent", "M4", 1, "stride", "conv2d",
       "16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D",
       "with-in mdspan extent mutated to 0 on a strided 7x7 view",
       ("with {i, j} in [Hpad/2, Wpad/2]",
        "with {i, j} in [0, Wpad/2]")),
    _m("M4.1.cv18.zeroextent", "M4", 1, "stride", "conv2d",
       "18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D",
       "with-in mdspan extent mutated to 0 on the down-sampled view",
       ("with {i, j} in [Hpad, Wpad]", "with {i, j} in [0, Wpad]")),
    _m("M4.1.cv21.zeroextent", "M4", 1, "stride", "conv2d",
       "21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D",
       "with-in mdspan extent mutated to 0 on the patchified-input view",
       ("with {i, j} in [Hpad, Wpad]", "with {i, j} in [0, Wpad]")),
    _m("M4.1.cv6.zeroextent", "M4", 1, "stride", "conv2d",
       "6_dynamic_64x64x56x56_128x64x3x3_64x128x56x56_S_P_D",
       "with-in mdspan extent mutated to 0 on a 3x3 dynamic-shape case",
       ("with {i, j} in [Hpad, Wpad]", "with {i, j} in [0, Wpad]")),
    _m("M4.1.cv8.zeroextent", "M4", 1, "stride", "conv2d",
       "8_dynamic_64x256x28x28_Cx256x3x3_64xCx28x28_S_P_D",
       "with-in mdspan extent mutated to 0 on a symbolic-channel case",
       ("with {i, j} in [Hpad, Wpad]", "with {i, j} in [0, Wpad]")),
    _m("M4.1.cv19.zeroextent", "M4", 1, "stride", "conv2d",
       "19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1",
       "with-in mdspan extent mutated to 0 on the 14x14 patch grid",
       ("with {x, y} in [Ho/14, Wo]", "with {x, y} in [0, Wo]")),
    _m("M4.1.cv1.zeroextent", "M4", 1, "stride", "conv2d",
       _C1, "element-loop extent mutated to 0 on the attention 1x1 path",
       ("foreach n in [N / #p]", "foreach n in [0]")),
    _m("M4.1.cv2.zeroextent", "M4", 1, "stride", "conv2d",
       "2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D",
       "outer tile extent mutated to 0 on a dynamic-batch case",
       ("foreach {co, n} in [Cout, N / #p / #q]",
        "foreach {co, n} in [0, N / #p / #q]")),
    _m("M4.1.rl2.zeroextent", "M4", 1, "stride", "relu",
       "2_cnn_128x128x28x28_128x128x28x28",
       "element-loop extent mutated to 0 on a rank-4 relu",
       ("foreach {i, j, k, l} in [I / #p, J, K, 7]",
        "foreach {i, j, k, l} in [0, J, K, 7]")),
    _m("M4.1.tp1.zeroextent", "M4", 1, "stride", "transpose",
       "1_bert_32x512x768_32x768x512",
       "transpose element-loop extent mutated to 0",
       ("foreach {y, z} in [512, 12]", "foreach {y, z} in [0, 12]")),
    _m("M4.1.tp14.zeroextent", "M4", 1, "stride", "transpose",
       "14_efficientnet_64x1280x7x7_64x7x7x1280",
       "extent mutated to 0 on the span-derived leading dimension",
       ("foreach {x, y} in [i.span(0)/#p/#q, i.span(1)]",
        "foreach {x, y} in [0, i.span(1)]")),
]

M4 += [
    # M4.2 -- a parallel-by bound mutated to 0. ZERO ONLY: the negative cases
    # live under M4.6, because "the space is legal but empty" and "the bound is
    # invalid" are different defects with different oracle expectations.
    _m("M4.2.cv11.zero", "M4", 2, "stride", "conv2d",
       "11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1",
       "parallelby bound mutated to 0 on a static 1x1 case",
       ("parallel q by 8  {", "parallel q by 0  {")),
    _m("M4.2.cv13.zero", "M4", 2, "stride", "conv2d",
       "13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1",
       "parallelby bound mutated to 0 where the bound is not a power of two",
       ("parallel q by 7  {", "parallel q by 0  {")),
    _m("M4.2.cv19.zero", "M4", 2, "stride", "conv2d",
       "19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1",
       "parallelby bound mutated to 0 on the patch-grid conv",
       ("parallel q by 14  {", "parallel q by 0  {")),
    _m("M4.6.cv2.neg", "M4", 2, "stride", "conv2d",
       "2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D",
       "parallelby bound mutated to negative on a dynamic-batch case",
       ("parallel q by 4  {", "parallel q by -4  {"), spec_id="M4.6"),
    _m("M4.2.cv3.zero", "M4", 2, "stride", "conv2d",
       "3_dynamic_16x256xHxW_256x256x3x3_16x256xHxW_S_P_D",
       "parallelby bound mutated to 0 where every extent is symbolic",
       ("parallel q by 4  {", "parallel q by 0  {")),
    _m("M4.2.cv16.zero", "M4", 2, "stride", "conv2d",
       "16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D",
       "OUTER parallelby bound mutated to 0, so the whole block goes away",
       ("parallel p by 1  {", "parallel p by 0  {")),
    _m("M4.6.rl2.neg", "M4", 2, "stride", "relu",
       "2_cnn_128x128x28x28_128x128x28x28",
       "parallelby bound mutated to negative on a rank-4 relu",
       ("parallel q by 4", "parallel q by -4"), spec_id="M4.6"),
    _m("M4.2.rl9.zero", "M4", 2, "stride", "relu",
       "9_dynamic_64x128xHxW_64x128xHxW",
       "parallelby bound mutated to 0 on a dynamic-shape relu",
       ("parallel q by 2", "parallel q by 0")),
    _m("M4.2.rl19.zero", "M4", 2, "stride", "relu",
       "19_transformer_32x512x2048_32x512x2048",
       "parallelby bound mutated to 0 on the largest relu case",
       ("parallel q by 64", "parallel q by 0")),
    _m("M4.6.tp1.neg", "M4", 2, "stride", "transpose",
       "1_bert_32x512x768_32x768x512",
       "parallelby bound mutated to negative on the transpose inner loop",
       ("parallel q by 64", "parallel q by -64"), spec_id="M4.6"),
    _m("M4.2.tp14.zero", "M4", 2, "stride", "transpose",
       "14_efficientnet_64x1280x7x7_64x7x7x1280",
       "parallelby bound mutated to 0 on the dma.transp case",
       ("parallel q by 4  {", "parallel q by 0  {")),
    _m("M4.2.tp17.zero", "M4", 2, "stride", "transpose",
       "17_mobilenet_128xx96x112x112_128x112x112x96",
       "parallelby bound mutated to 0 on a rank-5 transpose",
       ("parallel q by 16", "parallel q by 0")),
]

M4 += [
    # M4.3 -- a bound that is symbolic and vanishes only at runtime.
    _m("M4.3.cv1.symzero", "M4", 3, "stride", "conv2d",
       _C1, "loop bound spelled N - N: statically a real bound, zero at "
            "runtime",
       ("foreach n in [N / #p]", "foreach n in [N - N]")),
    _m("M4.3.cv4.symzero", "M4", 3, "stride", "conv2d",
       "4_dynamic_32x128xHxW_256x128x3x3_32x256xHxW_S_P_D",
       "channel loop bound cancelled to zero at runtime",
       ("foreach {co, n} in [Cout, N / #p / #q]",
        "foreach {co, n} in [Cout - Cout, N / #p / #q]")),
    _m("M4.3.cv5.symzero", "M4", 3, "stride", "conv2d",
       "5_dynamic_128xCx112x112_64xCx1x1_128x64x112x112_1_0_1",
       "symbolic batch loop bound that vanishes at runtime",
       ("foreach n in [N / #p]", "foreach n in [N - N]")),
    _m("M4.3.rl2.symzero", "M4", 3, "stride", "relu",
       "2_cnn_128x128x28x28_128x128x28x28",
       "symbolic leading bound cancelled on a rank-4 relu",
       ("foreach {i, j, k, l} in [I / #p, J, K, 7]",
        "foreach {i, j, k, l} in [I - I, J, K, 7]")),
    _m("M4.3.tp11.symzero", "M4", 3, "stride", "transpose",
       "11_dynamic_32xSx768_32x768xS",
       "symbolic sequence-length bound cancelled on the transpose path",
       ("foreach {y, z} in [seq_len, 12]",
        "foreach {y, z} in [seq_len - seq_len, 12]")),
]

M4 += [
    # M4.5 -- stride/step zero: every iteration maps onto the same tile.
    _m("M4.5.tp2.zerostep", "M4", 5, "stride", "transpose",
       "2_cnn_128x128x28x28_128x28x28x128",
       "output index of the transposed store multiplied by 0: all "
       "iterations write row 0",
       ("o.chunkat(p, z, w, y);", "o.chunkat(p, z, w, y * 0);")),
    _m("M4.5.tp11.zerostep", "M4", 5, "stride", "transpose",
       "11_dynamic_32xSx768_32x768xS",
       "transposed output index multiplied by 0 on the dynamic path",
       ("o.chunkat(p, z, y);", "o.chunkat(p, z, y * 0);")),
    _m("M4.5.tp12.zerostep", "M4", 5, "stride", "transpose",
       "12_dynamic_64xSx256_Sx64x256",
       "leading transposed output index multiplied by 0",
       ("o.chunkat(y, p, z);", "o.chunkat(y * 0, p, z);")),
    _m("M4.5.rl2.zerostep", "M4", 5, "stride", "relu",
       "2_cnn_128x128x28x28_128x128x28x28",
       "element index of the relu write-back multiplied by 0, so a single "
       "output element is overwritten N times",
       ("out.chunkat(p#i, j, k, l);", "out.chunkat(p#i, j, k, l * 0);")),
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

# ---- M3.17-M3.26 target limits: inadmissible by construction -------------
# One operator for the strongest cell (M3.20, block extent not a multiple of
# 32) so the launch-status path is exercised end to end; the rest stay
# `pending`/`na` and are listed by the GATE 2 cross-check.
M3 += [
    _m("M3.20.cv1.block3", "M3", 20, "hw", "conv2d",
       _C1,
       "block extent not a multiple of 32: a target-limit failure, not an "
       "obligation -- exercises the observation channel and is excluded from "
       "every admissible denominator",
       ("parallel p by 2  {", "parallel p by 3  {"), spec_id="M3.20"),
]

# ---------------------------------------------------------------------------
# Assemble the class cells. `Mut.cls` is the authority, NOT the list.
# ---------------------------------------------------------------------------
# M1.6 and M1.7 live in the `M1` list because that is the surface they edit,
# but they count in M4's cell (families M4-d and M4-e). Grouping by list
# membership -- which is what `ALL = {"M1": M1, ...}` used to do -- put 13
# M4 operators into M1's cell and left M4's cell looking short, while `cls`,
# the field that was supposed to say otherwise, was written and never read.
# Grouping by `cls` makes that field load-bearing; the assertion ties it to
# the registry so the two cannot drift apart again in silence.
def _axis_classes():
    """The mutation classes, from the one definition: `schema/class_axis.py`.

    Restating the class ids as a literal is what guard
    `axis.no-restated-tuples` forbids, and the old
    `ALL = {"M1": M1, "M2": M2, "M3": M3}` dodged that guard only by being a
    dict rather than a quoted tuple -- it hard-coded the same 3-class dialect
    the guard exists to catch. The axis is the authority, so ask it. Same
    lazy-import pattern as `choreo/stats.py`, for the same reason: this module
    is imported from several working directories.
    """
    import os as _os
    import sys as _sys
    _b2 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _b2 not in _sys.path:
        _sys.path.insert(0, _b2)
    from schema import class_axis as _AX
    return list(_AX.mutation_classes())


ALL = {c: [] for c in _axis_classes()}
for _x in M1 + M2 + M3 + M4:
    if _x.cls not in ALL:
        raise ValueError(
            "mutation %r declares class %r, which is not on the class axis "
            "%s" % (_x.id, _x.cls, sorted(ALL)))
    # Inline lookup rather than `class_of_spec()`: that helper is defined
    # further down, below the registry reconciliation, and this pass runs
    # first. `Mut.__init__` already refused an unregistered spec_id, so the
    # key is present.
    _want = SPEC_REGISTRY[_x.spec_id]["cls"]
    if _x.cls != _want:
        raise ValueError(
            "mutation %r is class %s but spec %s is registered as class %s: "
            "the operator and the registry disagree about which class cell "
            "this is evidence for" % (_x.id, _x.cls, _x.spec_id, _want))
    ALL[_x.cls].append(_x)


# ===========================================================================
# GATE 2 -- reconcile the registry against the operators that actually loaded.
#
# Doing this in code rather than by hand is the whole point: a hand-written
# `status=` drifts the moment an operator is added or removed, and a drifted
# registry is worse than no registry because it reads as coverage. After this
# block, SPEC_REGISTRY[sid]["status"] is exactly one of:
#
#   "implemented"  >=1 operator loaded for this spec
#   "na"           not applicable (avoided, or declared inadmissible) -- recorded,
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
    # `avoided` (the model repairs the state) and the noop controls are
    # inadmissible *by classification*, whether or not an operator exists --
    # which is why this cannot be folded into `status`:
    # M1.6/M3.6/M4.4/M3.17/M3.18/M3.20 are generated on purpose.
    if _meta["path"] == "avoided" or _sid in NOOP_BY_CONSTRUCTION:
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

# THE WORD `pending` NAMES TWO DIFFERENT SETS IN THIS MODULE -- defect D2.
#
#   status == "pending"        merely unwritten. Today 23 specs. This is what
#                              the record schema's enum value means.
#   `pending_specs`            unwritten AND admissible -- the GATE 2 list.
#                              Today EMPTY, because every unwritten spec is
#                              inadmissible, so the two sets are disjoint by
#                              accident rather than by design.
#
# `schema/PATCH-v2.1.md` glosses the enum value as "admissible, unwritten",
# which describes the *second* set, not the enum value. A reader who greps
# `pending` finds 23 where the gloss promises 0. The alias below exists so that
# any report can say which of the two questions it answered; the registry key
# `"pending"` is deliberately left alone, because renaming it would silently
# change the meaning of every record already written (the `L` -> M3 non-additive
# hazard) for no gain: no writer needs a third enum value, since an unwritten
# spec and an unwritten-admissible spec are both just "no operator yet".
admissible_unwritten_specs = pending_specs

# ---------------------------------------------------------------------------
# The method-family partition is NOT restated here.
# ---------------------------------------------------------------------------
# schema/method-taxonomy.json is the one definition of the family axis, and
# schema/method_taxonomy.py is its checker. The family is the level-2 unit that
# W1's budget N=8 is quoted per, so a lane that typed its own family names would
# drift exactly the way the class axis did before class-axis.json existed: a
# spec quietly re-homed to a neighbouring family still renders, so only a
# comparison against the one definition catches it.
#
# This is loaded rather than imported so that the taxonomy may add a family
# without touching this file -- the same reason class_axis is imported into
# stats.py instead of copied. The reverse direction (the taxonomy needs the
# registry) is a *parameter*, not an import, so there is no cycle.
_FAMILY_BY_SPEC = None


def family_by_spec():
    """{spec_id: family_id} over all 31 families, from the taxonomy."""
    global _FAMILY_BY_SPEC
    if _FAMILY_BY_SPEC is None:
        import json as _j
        import os as _o
        _root = _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))
        _p = _o.path.join(_root, "schema", "method-taxonomy.json")
        try:
            with open(_p) as _f:
                _d = _j.load(_f)
        except (IOError, OSError) as _e:
            raise ValueError(
                "cannot read the method taxonomy at %s (%s). It is the one "
                "definition of the family axis; a lane may not proceed "
                "without it." % (_p, _e))
        _m = {}
        for _fid, _fam in _d["families"].items():
            for _s in _fam.get("spec_ids", []):
                if _s in _m:
                    raise ValueError(
                        "spec %s is declared by two families (%s and %s). One "
                        "operator would then be evidence for two families and "
                        "count in two cells (defect D1)." % (_s, _m[_s], _fid))
                _m[_s] = _fid
        _FAMILY_BY_SPEC = _m
    return _FAMILY_BY_SPEC


def family_of(spec_id):
    """The family that owns `spec_id`, or None if it declares no family."""
    return family_by_spec().get(spec_id)


def check_family_partition():
    """Every registered spec is accounted for exactly once.

    The taxonomy is the definition and this is the consumer, so disagreement
    is always this file's or the taxonomy's bug -- never something to paper
    over.

    THREE buckets, not one. A spec can leave the family partition for exactly
    two legitimate reasons, and both are still *accounted for*:

      families            -- the spec is a realisation of some family
      attribution_only    -- launch-only; measured for attribution, never
                             enters a family's N
      p2_never_generated  -- structurally refused by the compiler, so it can
                             witness nothing

    `dead_declarations` is deliberately NOT a bucket here. It is a *state* a
    family member can be in (declared, no operator written yet) -- that is
    precisely the R_f<2 work item -- so treating it as an ownership bucket
    would let a family's only realisation be filed as "dead" and vanish from
    the denominator instead of showing up as a shortfall.

    Returns the number of specs checked; raises on any mismatch.
    """
    import json as _j
    import os as _o
    _root = _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))
    with open(_o.path.join(_root, "schema", "method-taxonomy.json")) as _f:
        _d = _j.load(_f)

    _attr = set()
    for _k in _d.get("attribution_only", {}).values():
        _attr |= set(_k.get("spec_ids", []))
    _p2 = set(_d.get("p2_never_generated", {}).get("spec_ids", []))

    m = family_by_spec()
    owned = set(m)
    unowned = sorted(s for s in SPEC_REGISTRY if s not in owned)
    unaccounted = sorted(s for s in unowned if s not in _attr and s not in _p2)
    if unaccounted:
        raise ValueError(
            "%d registered spec(s) belong to no family and are not declared "
            "attribution-only or avoided: %s. A spec in no bucket is a test "
            "outside the budget, so it can never be counted and never be "
            "missed." % (len(unaccounted), ", ".join(unaccounted)))

    # The other direction: a spec must not be filed in two buckets at once.
    # A launch-only operator that also claims to realise a family is how one
    # operator becomes evidence for two things.
    both = sorted(s for s in (_attr | _p2) if s in owned)
    if both:
        raise ValueError(
            "%d spec(s) are both a family realisation and filed as "
            "attribution-only / avoided: %s" % (len(both), ", ".join(both)))

    covered = len(owned) + len(_attr) + len(_p2)
    if covered != len(SPEC_REGISTRY):
        raise ValueError(
            "the three buckets cover %d specs but the registry has %d -- they "
            "must be disjoint and complete"
            % (covered, len(SPEC_REGISTRY)))
    return covered


def class_of_spec(spec_id):
    """The mutation class whose cell this spec's operators count in.

    Read from the REGISTRY, not from the id prefix. M1.6 and M1.7 are
    deliberately re-homed to class M4 (their edit is on an M1 index surface
    but the defect is M4's bound semantics), so for those two the prefix is a
    legacy label and the registry is the authority. Reading the prefix would
    put 13 M4-d/M4-e operators into M1's cell while leaving M4's cell looking
    empty -- a miscount that renders as a perfectly plausible number.
    """
    if spec_id not in SPEC_REGISTRY:
        raise ValueError("unregistered spec_id %r" % (spec_id,))
    return SPEC_REGISTRY[spec_id]["cls"]


def check_class_rehoming():
    """The two re-homed specs must keep their ids and their M4 class.

    Cheap guard against a well-meaning tidy-up "fixing" the M1.6 spelling and
    silently moving 13 operators back into M1's cell.
    """
    rehomed = {"M1.6": "M4", "M1.7": "M4"}
    for s, want in rehomed.items():
        got = SPEC_REGISTRY.get(s, {}).get("cls")
        if got != want:
            raise ValueError(
                "%s is declared class %s but must be %s: it is re-homed from "
                "M1 to M4 (its operators belong to family M4-d / M4-e, and "
                "counting them in M1's cell would inflate M1 while leaving "
                "M4's cell short)" % (s, got, want))
    return len(rehomed)


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
