#!/usr/bin/env python3
"""Mutation operator library for the choreo lane (E1).

Implements specs/mutation-specs.md §1-§3 as concrete source transforms on the
choreo reference surface.

Design note: transforms are keyed by (class, category, case), not by class
alone. The choreo suite is hand-written per case, so `scale.at(k)` exists in
layer_norm but not in transpose. A class-level table therefore silently skips
most (transform, case) pairs. Keying per case means every transform in this file
names source text that actually exists in the kernel it targets, and a skip is a
real defect in this file rather than an expected mismatch.

Ground rules (specs §0, §6, §7):
  * 3 classes: M1 element-access, M2 shape-compatibility, M3 hardware-constraint.
  * Iteration-validity is folded into M1 (spec 6: zero-stride / empty range).
  * A transform that does not change the source is a `noop` and is rejected.
  * N = 40 per class -> 160 total (specs §0). Counts are reached by enumerating
    defect *magnitudes* within a spec (e.g. K % 16 with residues 2/4/6/8/12),
    never by inventing defects outside §1-§3.
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
MINIMAL_SET = {
    "M1": ["layer_normalization", "softmax", "relu", "transpose"],
    "M2": ["layer_normalization", "matmul", "concat"],
    "M3": ["matmul", "conv2d"],
}

# Level-2 widening order (specs §5), applied only after level-1 is green.
LEVEL2_SET = {
    "M1": ["max_pool2d", "conv2d", "embedding", "batch_norm"],
    "M2": ["elemwise_add", "softmax"],
    "M3": ["batch_norm"],
}


class Mut:
    """One defect instance: an ordered list of literal textual edits.

    Each edit is (old, new, n) where n is the required occurrence count in the
    target source, or None meaning "at least one, replace all". Requiring an
    exact count is what stops a transform from rewriting sites it was not
    designed for.
    """

    def __init__(self, mid, cls, spec, paper_category, category, case, desc, edits):
        self.id = mid
        self.cls = cls
        self.spec = spec
        self.paper_category = paper_category
        self.category = category
        self.case = case
        self.desc = desc
        self.edits = edits


def _m(mid, cls, spec, pc, cat, case, desc, *edits):
    """Shorthand: edits are (old, new) or (old, new, n)."""
    norm = [e if len(e) == 3 else (e[0], e[1], None) for e in edits]
    return Mut(mid, cls, spec, pc, cat, case, desc, norm)


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
M3.append(_m("M3.s1.mm11.ktile5", "M3", 1, "dim-mismatch", "matmul",
             "11_dynamic_32xSx768_768x768_32xSx768",
             "k-tile count 5 yields a contraction tile that is not atom-divisible",
             ("with index = {m_tile, n_tile, k_tile} in [1, 6, 6] {",
              "with index = {m_tile, n_tile, k_tile} in [1, 6, 5] {")))
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
    cats = list(MINIMAL_SET[cls])
    if level2:
        cats += [c for c in LEVEL2_SET.get(cls, []) if c not in cats]
    return cats
