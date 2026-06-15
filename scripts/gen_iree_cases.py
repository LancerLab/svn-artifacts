#!/usr/bin/env python3
"""Generate IREE-compatible MLIR benchmarks using linalg-on-tensors.

Each function uses:
  - linalg.generic / named linalg ops (matmul, batch_matmul, transpose, etc.)
  - tensor<> types with dynamic ('?') dims where needed
  - No cf.assert (IREE does not support it)
  - Compatible with: iree-compile --iree-input-type=none

Output: benchmark/mlir/iree-cases/{category}/{stem}.mlir
"""

import re
import argparse
from pathlib import Path

ROOT       = Path(__file__).parent.parent
CHOREO_DIR = ROOT / "benchmark/choreo"
IREE_DIR   = ROOT / "benchmark/mlir/iree-cases"

# ---------------------------------------------------------------------------
# Shape utilities (mirrored from gen_mlir_cases.py)
# ---------------------------------------------------------------------------

def _to_dim(s):
    try:
        return int(s)
    except ValueError:
        return "?"

def parse_shape(s):
    return [_to_dim(p) for p in s.split("x") if p]

def tensor_type(dims, dtype="f32"):
    return "tensor<" + "x".join(str(d) for d in dims) + "x" + dtype + ">"

def _parts_after_index_name(stem, category):
    parts = stem.split("_")
    idx = 1
    name_parts = []
    while idx < len(parts):
        p = parts[idx]
        if p.isalpha() and p.islower():
            name_parts.append(p); idx += 1; continue
        if "x" in p: break
        if p and (p[0].isdigit() or p[0].isupper()): break
        name_parts.append(p); idx += 1
    return "_".join(name_parts), parts[idx:]

def shapes_for_category(stem, category):
    name, raw = _parts_after_index_name(stem, category)
    if category == "conv2d":
        shapes = [parse_shape(r) for r in raw if "x" in r]
        return shapes[:3]
    if category == "max_pool2d":
        shapes = [parse_shape(r) for r in raw]
        return [s for s in shapes if len(s) >= 2 or (len(s) == 1 and s[0] != "?")]
    return [parse_shape(r) for r in raw]

def find_reduce_dims(inp, out):
    used = [False] * len(inp)
    extra_out = []
    for oi, od in enumerate(out):
        matched = False
        for i, (id_, u) in enumerate(zip(inp, used)):
            if not u and id_ == od:
                used[i] = True; matched = True; break
        if not matched:
            for i, (id_, u) in enumerate(zip(inp, used)):
                if not u and (id_ == "?" or od == "?"):
                    used[i] = True; matched = True; break
        if not matched:
            extra_out.append((oi, od))
    return [i for i, u in enumerate(used) if not u], extra_out

def _find_concat_axis(inputs, C):
    nd = len(C)
    for d in range(nd):
        c_d = C[d]; in_d = [inp[d] for inp in inputs]
        if isinstance(c_d, int) and all(isinstance(x, int) for x in in_d):
            if sum(in_d) == c_d: return d
        if c_d == "?" and any(x == "?" for x in in_d):
            ok = True
            for d2 in range(nd):
                if d2 == d: continue
                sv = [v for v in [inp[d2] for inp in inputs] + [C[d2]] if isinstance(v, int)]
                if len(set(sv)) > 1: ok = False; break
            if ok: return d
    return nd - 1

def _find_expand_reassoc(A, C):
    ndA, ndC = len(A), len(C)
    def helper(ai, ci):
        if ai == ndA: return [] if ci == ndC else None
        max_size = ndC - ci - (ndA - ai - 1)
        for size in range(1, max_size + 1):
            grp = list(range(ci, ci + size))
            a_dim = A[ai]
            if a_dim == "?":
                if size == 1 and C[ci] == "?":
                    rest = helper(ai + 1, ci + 1)
                    if rest is not None: return [grp] + rest
            else:
                prod = 1; has_dyn = False
                for g in grp:
                    d = C[g]
                    if d == "?": has_dyn = True; break
                    prod *= d
                if not has_dyn and prod == a_dim:
                    rest = helper(ai + 1, ci + size)
                    if rest is not None: return [grp] + rest
        return None
    return helper(0, 0)

# ---------------------------------------------------------------------------
# IR helpers
# ---------------------------------------------------------------------------

def build_mlir(fname, args, body_lines, ret_dims, ret_dtype="f32"):
    if fname and fname[0].isdigit():
        fname = "f_" + fname
    arg_strs = []
    for entry in args:
        n, d = entry[0], entry[1]
        dt = entry[2] if len(entry) > 2 else "f32"
        arg_strs.append("%{}: {}".format(n, tensor_type(d, dt)))
    ret_t = tensor_type(ret_dims, ret_dtype)
    header = "  func.func @{}({}) -> {} {{".format(fname, ", ".join(arg_strs), ret_t)
    lines = ["module {", header] + body_lines + ["  }", "}"]
    return "\n".join(lines) + "\n"

def _consts(n):
    return ["    %c{} = arith.constant {} : index".format(i, i) for i in range(n)]

def _dims(argname, dims, dtype="f32"):
    t = tensor_type(dims, dtype)
    lines, dvars = [], {}
    for i in range(len(dims)):
        v = "%{}_d{}".format(argname, i)
        lines.append("    {} = tensor.dim %{}, %c{} : {}".format(v, argname, i, t))
        dvars[i] = v
    return lines, dvars

def _empty(dims, dv, dtype="f32", name="%out"):
    t = tensor_type(dims, dtype)
    dyn = [dv[i] for i, d in enumerate(dims) if d == "?"]
    args = "({})".format(", ".join(dyn)) if dyn else "()"
    return "    {} = tensor.empty{} : {}".format(name, args, t)

# ---------------------------------------------------------------------------
# linalg helpers
# ---------------------------------------------------------------------------

def _amap(n_iters, dim_indices):
    """Return affine_map<(d0,...,d_{n-1}) -> (d_i,...)> string."""
    domain = "({})".format(", ".join("d{}".format(i) for i in range(n_iters)))
    range_ = "({})".format(", ".join("d{}".format(i) for i in dim_indices))
    return "affine_map<{} -> {}>".format(domain, range_)

def _linalg_fill(result_name, fill_val_var, out_var, out_type):
    """linalg.fill to initialise a tensor."""
    return [
        "    {} = linalg.fill ins({} : f32) outs({} : {}) -> {}".format(
            result_name, fill_val_var, out_var, out_type, out_type),
    ]

def _lg(res, n, ins_specs, outs_specs, iter_types, maps, region):
    """
    Build a linalg.generic block.
    ins_specs / outs_specs: list of "varname : type" strings
    maps: list of dimension index lists (one per ins tensor + one per outs tensor)
    region: list of lines inside the { ^bb0 ... linalg.yield ... } region
    Uses linalg multi-operand format: ins(%a, %b : typeA, typeB).
    """
    aff_maps = "[{}]".format(", ".join(_amap(n, d) for d in maps))
    itypes   = "[{}]".format(", ".join('"{}"'  .format(t) for t in iter_types))
    if ins_specs:
        in_vars  = ", ".join(s.split(":")[0].strip() for s in ins_specs)
        in_types = ", ".join(":".join(s.split(":")[1:]).strip() for s in ins_specs)
        ins_str  = "{} : {}".format(in_vars, in_types)
    else:
        ins_str = ""
    if outs_specs:
        out_vars  = ", ".join(s.split(":")[0].strip() for s in outs_specs)
        out_types = ", ".join(":".join(s.split(":")[1:]).strip() for s in outs_specs)
        outs_str  = "{} : {}".format(out_vars, out_types)
    else:
        outs_str = ""
    res_type_list = [":".join(s.split(":")[1:]).strip() for s in outs_specs]
    if len(res_type_list) == 1:
        res_types = res_type_list[0]
    else:
        res_types = "({})".format(", ".join(res_type_list))
    lines = [
        "    {} = linalg.generic {{".format(res),
        "      indexing_maps = {},".format(aff_maps),
        "      iterator_types = {}".format(itypes),
        "    }} ins({}) outs({}) {{".format(ins_str, outs_str),
    ] + region + [
        "    }} -> {}".format(res_types),
    ]
    return lines

# ---------------------------------------------------------------------------
# Category generators
# ---------------------------------------------------------------------------

# ---- unary pointwise: relu, sigmoid, gelu ----

def _gen_unary_linalg(stem, shapes, region_body):
    """Pointwise unary op via linalg.generic."""
    if not shapes:
        return None
    A = shapes[0]; nd = len(A); fname = stem
    tA = tensor_type(A)
    body = _consts(max(2, nd))
    a_lines, a_d = _dims("input", A)
    body += a_lines
    body.append(_empty(A, {i: a_d[i] for i in range(nd) if A[i] == "?"}, name="%out"))
    maps   = [list(range(nd)), list(range(nd))]
    itype  = ["parallel"] * nd
    region = [
        "    ^bb0(%in: f32, %init: f32):",
    ] + ["      " + l for l in region_body]
    body += _lg("%result", nd, ["%input : {}".format(tA)], ["%out : {}".format(tA)],
                itype, maps, region)
    body.append("    return %result : {}".format(tA))
    return build_mlir(fname, [("input", A)], body, A)

def gen_relu(stem, shapes):
    return _gen_unary_linalg(stem, shapes, [
        "%zero = arith.constant 0.0 : f32",
        "%res = arith.maximumf %in, %zero : f32",
        "linalg.yield %res : f32",
    ])

def gen_sigmoid(stem, shapes):
    return _gen_unary_linalg(stem, shapes, [
        "%neg = arith.negf %in : f32",
        "%expv = math.exp %neg : f32",
        "%one = arith.constant 1.0 : f32",
        "%den = arith.addf %one, %expv : f32",
        "%res = arith.divf %one, %den : f32",
        "linalg.yield %res : f32",
    ])

def gen_gelu(stem, shapes):
    return _gen_unary_linalg(stem, shapes, [
        "%half   = arith.constant 0.5 : f32",
        "%isqrt2 = arith.constant 0.7071067811865476 : f32",
        "%sc     = arith.mulf %in, %isqrt2 : f32",
        "%efv    = math.erf %sc : f32",
        "%one    = arith.constant 1.0 : f32",
        "%ep1    = arith.addf %efv, %one : f32",
        "%hx     = arith.mulf %in, %half : f32",
        "%res    = arith.mulf %hx, %ep1 : f32",
        "linalg.yield %res : f32",
    ])

# ---- elemwise_add ----

def gen_elemwise_add(stem, shapes):
    if len(shapes) < 3:
        return None
    A, B, C = shapes[0], shapes[1], shapes[2]
    fname = stem; ndA = len(A); ndB = len(B); ndC = len(C)
    tA = tensor_type(A); tB = tensor_type(B); tC = tensor_type(C)
    body = _consts(max(2, ndA, ndB, ndC))
    a_lines, a_d = _dims("lhs", A)
    b_lines, b_d = _dims("rhs", B)
    body += a_lines + b_lines
    out_dv = {i: a_d[i] for i in range(ndC) if C[i] == "?"}
    body.append(_empty(C, out_dv, name="%out"))

    if ndA == ndB:
        # Same-rank: simple identity maps for both inputs
        maps   = [list(range(ndC)), list(range(ndC)), list(range(ndC))]
        itype  = ["parallel"] * ndC
        region = [
            "    ^bb0(%a: f32, %b: f32, %init: f32):",
            "      %res = arith.addf %a, %b : f32",
            "      linalg.yield %res : f32",
        ]
        body += _lg("%result", ndC,
                    ["%lhs : {}".format(tA), "%rhs : {}".format(tB)],
                    ["%out : {}".format(tC)],
                    itype, maps, region)
    elif ndB == 1 and B[0] == 1:
        # B is scalar (tensor<1xf32>): extract and add elementwise
        body.append("    %c0b = arith.constant 0 : index")
        body.append("    %scalar_b = tensor.extract %rhs[%c0b] : {}".format(tB))
        maps   = [list(range(ndC)), list(range(ndC))]
        itype  = ["parallel"] * ndC
        region = [
            "    ^bb0(%a: f32, %init: f32):",
            "      %res = arith.addf %a, %scalar_b : f32",
            "      linalg.yield %res : f32",
        ]
        body += _lg("%result", ndC,
                    ["%lhs : {}".format(tA)],
                    ["%out : {}".format(tC)],
                    itype, maps, region)
    elif ndB == 1:
        # B is 1-D (broadcast over all dims except the matching one)
        # Find which dimension of A/C matches B[0]
        b_match = ndC - 1
        for j, cv in enumerate(C):
            if cv == B[0]:
                b_match = j; break
        maps   = [list(range(ndC)), [b_match], list(range(ndC))]
        itype  = ["parallel"] * ndC
        region = [
            "    ^bb0(%a: f32, %b: f32, %init: f32):",
            "      %res = arith.addf %a, %b : f32",
            "      linalg.yield %res : f32",
        ]
        body += _lg("%result", ndC,
                    ["%lhs : {}".format(tA), "%rhs : {}".format(tB)],
                    ["%out : {}".format(tC)],
                    itype, maps, region)
    else:
        # Fall back: same shape assumption
        maps   = [list(range(ndC)), list(range(ndC)), list(range(ndC))]
        itype  = ["parallel"] * ndC
        region = [
            "    ^bb0(%a: f32, %b: f32, %init: f32):",
            "      %res = arith.addf %a, %b : f32",
            "      linalg.yield %res : f32",
        ]
        body += _lg("%result", ndC,
                    ["%lhs : {}".format(tA), "%rhs : {}".format(tB)],
                    ["%out : {}".format(tC)],
                    itype, maps, region)

    body.append("    return %result : {}".format(tC))
    return build_mlir(fname, [("lhs", A), ("rhs", B)], body, C)

# ---- softmax ----

def gen_softmax(stem, shapes):
    if not shapes:
        return None
    A = shapes[0]; nd = len(A); fname = stem
    tA = tensor_type(A)
    # outer dims (all but last), inner (last) dim
    outer = A[:-1]; ndO = len(outer)
    tOuter = tensor_type(outer) if outer else "tensor<f32>"

    body = _consts(max(2, nd))
    a_lines, a_d = _dims("input", A)
    body += a_lines
    # output for softmax has same shape as input
    body.append(_empty(A, {i: a_d[i] for i in range(nd) if A[i] == "?"}, name="%out"))

    # ---- Pass 1: max reduction over last dim ----
    outer_dv = {i: a_d[i] for i in range(ndO) if outer[i] == "?"} if outer else {}
    body.append(_empty(outer, outer_dv, name="%max_t") if outer
                else "    %max_t = tensor.empty() : tensor<f32>")
    body.append("    %neg_inf = arith.constant -3.4028234663852886e+38 : f32")
    body += _linalg_fill("%max_init", "%neg_inf", "%max_t", tOuter)

    # maps: input -> full range, max -> outer dims, iter_types with last=reduction
    maps_p1  = [list(range(nd)), list(range(ndO))]
    itypes_p1 = ["parallel"] * ndO + ["reduction"]
    region_p1 = [
        "    ^bb0(%in: f32, %cur: f32):",
        "      %mx = arith.maximumf %in, %cur : f32",
        "      linalg.yield %mx : f32",
    ]
    body += _lg("%max_result", nd,
                ["%input : {}".format(tA)],
                ["%max_init : {}".format(tOuter)],
                itypes_p1, maps_p1, region_p1)

    # ---- Pass 2: sum of exp(x - max) over last dim ----
    body.append(_empty(outer, outer_dv, name="%sum_t") if outer
                else "    %sum_t = tensor.empty() : tensor<f32>")
    body.append("    %zero_f = arith.constant 0.0 : f32")
    body += _linalg_fill("%sum_init", "%zero_f", "%sum_t", tOuter)

    maps_p2  = [list(range(nd)), list(range(ndO)), list(range(ndO))]
    itypes_p2 = ["parallel"] * ndO + ["reduction"]
    region_p2 = [
        "    ^bb0(%in: f32, %mx: f32, %acc: f32):",
        "      %sh  = arith.subf %in, %mx : f32",
        "      %ex  = math.exp %sh : f32",
        "      %nac = arith.addf %acc, %ex : f32",
        "      linalg.yield %nac : f32",
    ]
    body += _lg("%sum_result", nd,
                ["%input : {}".format(tA), "%max_result : {}".format(tOuter)],
                ["%sum_init : {}".format(tOuter)],
                itypes_p2, maps_p2, region_p2)

    # ---- Pass 3: normalise (all parallel) ----
    maps_p3  = [list(range(nd)), list(range(ndO)), list(range(ndO)), list(range(nd))]
    itypes_p3 = ["parallel"] * nd
    region_p3 = [
        "    ^bb0(%in: f32, %mx: f32, %sm: f32, %init: f32):",
        "      %sh  = arith.subf %in, %mx : f32",
        "      %ex  = math.exp %sh : f32",
        "      %res = arith.divf %ex, %sm : f32",
        "      linalg.yield %res : f32",
    ]
    body += _lg("%result", nd,
                ["%input : {}".format(tA),
                 "%max_result : {}".format(tOuter),
                 "%sum_result : {}".format(tOuter)],
                ["%out : {}".format(tA)],
                itypes_p3, maps_p3, region_p3)

    body.append("    return %result : {}".format(tA))
    return build_mlir(fname, [("input", A)], body, A)

# ---- matmul ----

def gen_matmul(stem, shapes):
    if len(shapes) < 3:
        return None
    A, B, C = shapes[0], shapes[1], shapes[2]
    fname = stem; ndA = len(A); ndB = len(B)
    tA = tensor_type(A); tB = tensor_type(B); tC = tensor_type(C)
    body = _consts(max(2, max(ndA, ndB, len(C))))
    a_lines, a_d = _dims("lhs", A)
    b_lines, b_d = _dims("rhs", B)
    body += a_lines + b_lines
    body.append("    %zero = arith.constant 0.0 : f32")

    if ndA == 2 and ndB == 2:
        # linalg.matmul: C[m,n] += A[m,k] * B[k,n]
        out_dv = {}
        if C[0] == "?": out_dv[0] = a_d[0]
        if C[1] == "?": out_dv[1] = b_d[1]
        body.append(_empty(C, out_dv, name="%out"))
        body += _linalg_fill("%filled", "%zero", "%out", tC)
        body.append("    %result = linalg.matmul"
                    " ins(%lhs, %rhs : {}, {})"
                    " outs(%filled : {}) -> {}".format(tA, tB, tC, tC))

    elif ndA == 3 and ndB == 3:
        # linalg.batch_matmul: C[b,m,n] += A[b,m,k] * B[b,k,n]
        out_dv = {i: a_d[i] for i in range(3) if C[i] == "?"}
        if C[2] == "?": out_dv[2] = b_d[2]
        body.append(_empty(C, out_dv, name="%out"))
        body += _linalg_fill("%filled", "%zero", "%out", tC)
        body.append("    %result = linalg.batch_matmul"
                    " ins(%lhs, %rhs : {}, {})"
                    " outs(%filled : {}) -> {}".format(tA, tB, tC, tC))

    elif ndA == 3 and ndB == 2:
        # Batched 3D x 2D: C[bs,m,n] += A[bs,m,k] * B[k,n]  (linalg.generic)
        out_dv = {i: a_d[i] for i in range(2) if C[i] == "?"}
        if C[2] == "?": out_dv[2] = b_d[1]
        body.append(_empty(C, out_dv, name="%out"))
        body += _linalg_fill("%filled", "%zero", "%out", tC)
        # n_iters=4: d0=bs, d1=m, d2=n, d3=k
        maps   = [[0,1,3], [3,2], [0,1,2]]
        itypes = ["parallel","parallel","parallel","reduction"]
        region = [
            "    ^bb0(%a: f32, %b: f32, %acc: f32):",
            "      %p = arith.mulf %a, %b : f32",
            "      %r = arith.addf %acc, %p : f32",
            "      linalg.yield %r : f32",
        ]
        body += _lg("%result", 4,
                    ["%lhs : {}".format(tA), "%rhs : {}".format(tB)],
                    ["%filled : {}".format(tC)],
                    itypes, maps, region)

    elif ndA == 4 and ndB == 4:
        # Double-batch: C[b0,b1,m,n] += A[b0,b1,m,k] * B[b0,b1,k,n]
        out_dv = {i: a_d[i] for i in range(3) if C[i] == "?"}
        if C[3] == "?": out_dv[3] = b_d[3]
        body.append(_empty(C, out_dv, name="%out"))
        body += _linalg_fill("%filled", "%zero", "%out", tC)
        # n_iters=5: d0=b0, d1=b1, d2=m, d3=n, d4=k
        maps   = [[0,1,2,4], [0,1,4,3], [0,1,2,3]]
        itypes = ["parallel","parallel","parallel","parallel","reduction"]
        region = [
            "    ^bb0(%a: f32, %b: f32, %acc: f32):",
            "      %p = arith.mulf %a, %b : f32",
            "      %r = arith.addf %acc, %p : f32",
            "      linalg.yield %r : f32",
        ]
        body += _lg("%result", 5,
                    ["%lhs : {}".format(tA), "%rhs : {}".format(tB)],
                    ["%filled : {}".format(tC)],
                    itypes, maps, region)
    else:
        return None

    body.append("    return %result : {}".format(tC))
    return build_mlir(fname, [("lhs", A), ("rhs", B)], body, C)

# ---- transpose ----

def gen_transpose(stem, shapes):
    if len(shapes) < 2:
        return None
    A, C = shapes[0], shapes[1]; nd = len(A); fname = stem
    tA = tensor_type(A); tC = tensor_type(C)
    # Compute permutation: for each dim of C, find its position in A
    used = [False] * nd; perm = []
    for cv in C:
        best = None
        for ai, av in enumerate(A):
            if not used[ai] and av == cv:
                best = ai; break
        if best is None:
            for ai in range(nd):
                if not used[ai]: best = ai; break
        perm.append(best); used[best] = True
    body = _consts(max(2, nd))
    a_lines, a_d = _dims("input", A)
    body += a_lines
    out_dv = {oi: a_d[perm[oi]] for oi in range(nd) if C[oi] == "?"}
    body.append(_empty(C, out_dv, name="%out"))
    perm_str = "[{}]".format(", ".join(str(p) for p in perm))
    body.append("    %result = linalg.transpose ins(%input : {}) outs(%out : {}) permutation = {}".format(
        tA, tC, perm_str))
    body.append("    return %result : {}".format(tC))
    return build_mlir(fname, [("input", A)], body, C)

# ---- reshape ----

def _reshape_expand(fname, A, C):
    reassoc = _find_expand_reassoc(A, C)
    if reassoc is None:
        return None
    reassoc_str = "[" + ", ".join("[" + ", ".join(str(x) for x in g) + "]" for g in reassoc) + "]"
    tA = tensor_type(A); tC = tensor_type(C)
    ndA = len(A)
    ci_to_ai = {}
    for ai, grp in enumerate(reassoc):
        for ci in grp: ci_to_ai[ci] = ai
    preamble = []
    a_d = {}
    if any(d == "?" for d in C):
        preamble += _consts(max(2, ndA))
        a_lines, a_d = _dims("input", A)
        preamble += a_lines
    out_tokens = []
    for ci, cd in enumerate(C):
        ai = ci_to_ai[ci]
        if cd == "?" and len(reassoc[ai]) == 1 and A[ai] == "?":
            out_tokens.append(a_d[ai])
        elif isinstance(cd, int):
            out_tokens.append(str(cd))
        else:
            return None
    out_shape_str = "[" + ", ".join(out_tokens) + "]"
    body = preamble + [
        "    %out = tensor.expand_shape %input {} output_shape {} : {} into {}".format(
            reassoc_str, out_shape_str, tA, tC),
        "    return %out : {}".format(tC),
    ]
    return build_mlir(fname, [("input", A)], body, C)

def _reshape_collapse(fname, A, C):
    ndA, ndC = len(A), len(C)
    reassoc = [[] for _ in range(ndC)]
    ci = 0
    for ai in range(ndA):
        if ci < ndC - 1 and A[ai] == C[ci]:
            reassoc[ci].append(ai); ci += 1
        else:
            reassoc[ci].append(ai)
    for i in range(ndC):
        if not reassoc[i]:
            if i > 0 and len(reassoc[i - 1]) > 1:
                reassoc[i].append(reassoc[i - 1].pop())
    for i, grp in enumerate(reassoc):
        grp_dims = [A[j] for j in grp]
        has_dyn = any(d == "?" for d in grp_dims); out_d = C[i]
        if has_dyn and isinstance(out_d, int): return None
        if not has_dyn:
            prod = 1
            for d in grp_dims: prod *= d
            if isinstance(out_d, int) and prod != out_d: return None
    reassoc_str = "[" + ", ".join("[" + ", ".join(str(x) for x in g) + "]" for g in reassoc) + "]"
    tA = tensor_type(A); tC = tensor_type(C)
    lines = [
        "    %out = tensor.collapse_shape %input {} : {} into {}".format(reassoc_str, tA, tC),
        "    return %out : {}".format(tC),
    ]
    return build_mlir(fname, [("input", A)], lines, C)

def gen_reshape(stem, shapes):
    if len(shapes) < 2:
        return None
    A, C = shapes[0], shapes[1]; fname = stem
    ndA, ndC = len(A), len(C)
    if ndC > ndA:
        return _reshape_expand(fname, A, C)
    elif ndC < ndA:
        return _reshape_collapse(fname, A, C)
    else:
        # Same rank - try collapse+expand trick or transpose
        # If same rank but different shape, it's a reshape with no clear assoc -
        # fall back to expand then collapse
        tA = tensor_type(A); tC = tensor_type(C)
        # Try 1: collapse to rank-1 then expand
        mid_size = 1
        mid_dyn = False
        for d in A:
            if d == "?": mid_dyn = True; break
            mid_size *= d
        if mid_dyn:
            return None  # Can't handle
        mid_dims = [mid_size]
        tMid = tensor_type(mid_dims)
        reassoc_coll = "[[{}]]".format(", ".join(str(i) for i in range(ndA)))
        reassoc_exp = _find_expand_reassoc(mid_dims, C)
        if reassoc_exp is None:
            return None
        reassoc_exp_str = "[" + ", ".join("[" + ", ".join(str(x) for x in g) + "]" for g in reassoc_exp) + "]"
        out_tokens = []
        for d in C:
            if isinstance(d, int): out_tokens.append(str(d))
            else: return None
        out_shape_str = "[" + ", ".join(out_tokens) + "]"
        body = [
            "    %mid = tensor.collapse_shape %input {} : {} into {}".format(reassoc_coll, tA, tMid),
            "    %out = tensor.expand_shape %mid {} output_shape {} : {} into {}".format(
                reassoc_exp_str, out_shape_str, tMid, tC),
            "    return %out : {}".format(tC),
        ]
        return build_mlir(fname, [("input", A)], body, C)

# ---- concat ----

def gen_concat(stem, shapes):
    if len(shapes) < 3:
        return None
    inputs = shapes[:-1]; C = shapes[-1]
    fname = stem; nd = len(C)
    axis = _find_concat_axis(inputs, C)

    body = _consts(max(2, nd))
    # For each input, only tensor.dim the dynamic dims; static dims use literals.
    in_sv = []  # in_sv[ii][d] = "32" (static) or "%in0_d1" (dynamic)
    for ii, inp in enumerate(inputs):
        sv = {}
        t_inp = tensor_type(inp)
        for d in range(nd):
            if inp[d] == "?":
                v = "%in{}_d{}".format(ii, d)
                body.append("    {} = tensor.dim %in{}, %c{} : {}".format(v, ii, d, t_inp))
                sv[d] = v
            else:
                sv[d] = str(inp[d])
        in_sv.append(sv)

    tC = tensor_type(C)
    # Compute output tensor empty args (only dynamic dims)
    out_dv = {}
    for d in range(nd):
        if C[d] == "?":
            if d != axis:
                out_dv[d] = in_sv[0][d]
            else:
                # Sum input axis dims
                acc = "%c0"
                for ii in range(len(inputs)):
                    av = in_sv[ii][axis]
                    tmp = "%csum{}".format(ii)
                    if av.isdigit() or av[0] == "-":
                        cv = "%axis_lit{}".format(ii)
                        body.append("    {} = arith.constant {} : index".format(cv, av))
                        av = cv
                    body.append("    {} = arith.addi {}, {} : index".format(tmp, acc, av))
                    acc = tmp
                out_dv[axis] = acc
    body.append(_empty(C, out_dv, name="%out"))

    # tensor.insert_slice for each input
    # Track cumulative axis offset as integer when all static, else as SSA var
    cum_int = 0    # integer offset (valid when cum_is_int==True)
    cum_var = None # SSA var for offset (valid when cum_is_int==False)
    cum_is_int = True

    carry = "%out"
    for ii, inp in enumerate(inputs):
        t_inp = tensor_type(inp)
        sv_ii = in_sv[ii]
        # sizes: literal for static dims, SSA var for dynamic dims
        size_parts = [sv_ii[d] for d in range(nd)]
        # offsets: 0 for non-axis dims; cumulative for axis
        off_parts = []
        for d in range(nd):
            if d != axis:
                off_parts.append("%c0")
            else:
                if cum_is_int:
                    if cum_int == 0:
                        off_parts.append("%c0")
                    else:
                        cv = "%coff{}".format(cum_int)
                        body.append("    {} = arith.constant {} : index".format(cv, cum_int))
                        off_parts.append(cv)
                else:
                    off_parts.append(cum_var)
        strides_str = "[{}]".format(", ".join("1" for _ in range(nd)))
        offs_str    = "[{}]".format(", ".join(off_parts))
        sizes_str   = "[{}]".format(", ".join(size_parts))
        next_carry  = "%ins{}".format(ii)
        body.append("    {} = tensor.insert_slice %in{} into {}{}{}{}".format(
            next_carry, ii, carry, offs_str, sizes_str, strides_str))
        body[-1] += " : {} into {}".format(t_inp, tC)
        carry = next_carry
        # Update cumulative axis offset
        if ii < len(inputs) - 1:
            av = sv_ii[axis]
            if cum_is_int and (av.isdigit() or (av.startswith("-") and av[1:].isdigit())):
                cum_int += int(av)
            else:
                # Switch to dynamic tracking
                if av.isdigit() or (av.startswith("-") and av[1:].isdigit()):
                    cv = "%axis_add{}".format(ii)
                    body.append("    {} = arith.constant {} : index".format(cv, av))
                    av = cv
                next_cum = "%cum{}".format(ii)
                if cum_is_int:
                    if cum_int == 0:
                        base = "%c0"
                    else:
                        base = "%cbase{}".format(cum_int)
                        body.append("    {} = arith.constant {} : index".format(base, cum_int))
                    body.append("    {} = arith.addi {}, {} : index".format(next_cum, base, av))
                else:
                    body.append("    {} = arith.addi {}, {} : index".format(next_cum, cum_var, av))
                cum_is_int = False
                cum_var = next_cum

    body.append("    return {} : {}".format(carry, tC))
    arg_list = [("in{}".format(ii), inp) for ii, inp in enumerate(inputs)]
    return build_mlir(fname, arg_list, body, C)

# ---- batch_norm ----

def gen_batch_norm(stem, shapes):
    """Simplified batch norm: out = input * gamma + beta (no running stats)."""
    if len(shapes) < 3:
        return None
    inp = shapes[0]; gamma_dims = shapes[1]; beta_dims = shapes[2] if len(shapes) >= 4 else shapes[1]
    C_out = shapes[-1]; nd = len(inp); fname = stem
    # Find the channel axis
    ch_axis = 1
    for ai, av in enumerate(inp):
        if av == gamma_dims[0]: ch_axis = ai; break
    tA = tensor_type(inp); tG = tensor_type(gamma_dims); tBt = tensor_type(beta_dims)
    tC = tensor_type(C_out)
    body = _consts(max(2, nd))
    a_lines, a_d = _dims("input", inp)
    body += a_lines
    out_dv = {i: a_d[i] for i in range(nd) if C_out[i] == "?"}
    body.append(_empty(C_out, out_dv, name="%out"))
    # linalg.generic: all parallel
    # maps: input=identity, gamma=[ch_axis], beta=[ch_axis], output=identity
    maps   = [list(range(nd)), [ch_axis], [ch_axis], list(range(nd))]
    itype  = ["parallel"] * nd
    region = [
        "    ^bb0(%x: f32, %g: f32, %b: f32, %init: f32):",
        "      %sc  = arith.mulf %x, %g : f32",
        "      %res = arith.addf %sc, %b : f32",
        "      linalg.yield %res : f32",
    ]
    body += _lg("%result", nd,
                ["%input : {}".format(tA), "%gamma : {}".format(tG), "%beta : {}".format(tBt)],
                ["%out : {}".format(tC)],
                itype, maps, region)
    body.append("    return %result : {}".format(tC))
    return build_mlir(fname, [("input", inp), ("gamma", gamma_dims), ("beta", beta_dims)], body, C_out)

# ---- layer_normalization ----

def gen_layer_norm(stem, shapes):
    """Layer norm via 3-pass linalg.generic (sum+sumsq / stats / normalise)."""
    if len(shapes) < 2:
        return None
    inp = shapes[0]; nd = len(inp)
    gamma_dims = shapes[1]; beta_dims = shapes[2] if len(shapes) >= 3 else gamma_dims
    norm_rank  = len(gamma_dims); batch_rank = max(0, nd - norm_rank)
    fname = stem

    tA  = tensor_type(inp)
    tG  = tensor_type(gamma_dims)
    tBt = tensor_type(beta_dims)

    outer_shape  = inp[:batch_rank]          # shape of accumulator tensors
    tOuter = tensor_type(outer_shape) if outer_shape else "tensor<f32>"

    body = _consts(max(2, nd))
    a_lines, a_d = _dims("input", inp)
    g_lines, g_d = _dims("gamma", gamma_dims)
    b_lines, b_d = _dims("beta",  beta_dims)
    body += a_lines + g_lines + b_lines

    out_dv = {i: a_d[i] for i in range(nd) if inp[i] == "?"}
    body.append(_empty(inp, out_dv, name="%out"))

    # Outer accumulator empty tensors
    outer_dv = {i: a_d[i] for i in range(batch_rank) if outer_shape[i] == "?"} if outer_shape else {}
    body.append(_empty(outer_shape, outer_dv, name="%sum_t") if outer_shape
                else "    %sum_t = tensor.empty() : tensor<f32>")
    body.append(_empty(outer_shape, outer_dv, name="%sumsq_t") if outer_shape
                else "    %sumsq_t = tensor.empty() : tensor<f32>")

    body.append("    %zero_f   = arith.constant 0.0 : f32")
    body += _linalg_fill("%sum_init",   "%zero_f", "%sum_t",   tOuter)
    body += _linalg_fill("%sumsq_init", "%zero_f", "%sumsq_t", tOuter)

    # Normalisation element count
    norm_total = 1; norm_dyn = False
    for ni in range(norm_rank):
        d = inp[batch_rank + ni]
        if isinstance(d, int): norm_total *= d
        else: norm_dyn = True; break
    if norm_dyn:
        body.append("    %nsz_idx_0 = arith.constant 1 : index")
        for ni in range(norm_rank):
            body.append("    %nsz_idx_{} = arith.muli %nsz_idx_{}, {} : index".format(
                ni + 1, ni, a_d[batch_rank + ni]))
        body.append("    %nsz_i64 = arith.index_cast %nsz_idx_{} : index to i64".format(norm_rank))
        body.append("    %nsz = arith.sitofp %nsz_i64 : i64 to f32")
    else:
        body.append("    %nsz = arith.constant {} : f32".format(float(norm_total)))

    # Pass 1: compute sum and sum-of-squares simultaneously
    # n_iters = nd;  iter d0..d_{batch-1} are parallel, d_batch..d_{nd-1} are reduction
    maps_p1 = [list(range(nd)),           # input: identity
               list(range(batch_rank)),   # sum output: outer dims
               list(range(batch_rank))]   # sumsq output: outer dims
    if not outer_shape:
        maps_p1 = [list(range(nd)), [], []]
    itypes_p1 = ["parallel"] * batch_rank + ["reduction"] * norm_rank

    body_p1 = [
        "    ^bb0(%in: f32, %sum_acc: f32, %sq_acc: f32):",
        "      %sq    = arith.mulf %in, %in : f32",
        "      %nsum  = arith.addf %sum_acc, %in : f32",
        "      %nsq   = arith.addf %sq_acc, %sq : f32",
        "      linalg.yield %nsum, %nsq : f32, f32",
    ]
    body += _lg("%sum_result, %sumsq_result", nd,
                ["%input : {}".format(tA)],
                ["%sum_init : {}".format(tOuter), "%sumsq_init : {}".format(tOuter)],
                itypes_p1, maps_p1, body_p1)

    body.append("    %eps = arith.constant 1.0e-05 : f32")

    # Pass 2: compute mean and inv_std tensors (element-wise over outer shape)
    body.append(_empty(outer_shape, outer_dv, name="%mean_t") if outer_shape
                else "    %mean_t = tensor.empty() : tensor<f32>")
    body.append(_empty(outer_shape, outer_dv, name="%istd_t") if outer_shape
                else "    %istd_t = tensor.empty() : tensor<f32>")
    body += _linalg_fill("%mean_dummy", "%zero_f", "%mean_t", tOuter)
    body += _linalg_fill("%istd_dummy", "%zero_f", "%istd_t", tOuter)

    n_outer = batch_rank
    maps_p2 = [list(range(n_outer)), list(range(n_outer)),  # sum, sumsq ins
               list(range(n_outer)), list(range(n_outer))]   # mean, istd outs
    if not outer_shape:
        maps_p2 = [[], [], [], []]
    itypes_p2 = ["parallel"] * n_outer if n_outer > 0 else ["parallel"]
    if n_outer == 0:
        # scalar case: compute outside linalg
        body += [
            "    %one_f  = arith.constant 1.0 : f32",
            "    %mean_s  = arith.divf %sum_result, %nsz : f32",    # placeholder extraction
            "    %msq_s   = arith.mulf %mean_s, %mean_s : f32",
            "    %esq_s   = arith.divf %sumsq_result, %nsz : f32",
            "    %var_s   = arith.subf %esq_s, %msq_s : f32",
            "    %vep_s   = arith.addf %var_s, %eps : f32",
            "    %std_s   = math.sqrt %vep_s : f32",
            "    %istd_s  = arith.divf %one_f, %std_s : f32",
            "    %mean_result = tensor.insert %mean_s into %mean_dummy[] : tensor<f32>",
            "    %istd_result = tensor.insert %istd_s into %istd_dummy[] : tensor<f32>",
        ]
    else:
        body_p2 = [
            "    ^bb0(%sm: f32, %sq: f32, %m_init: f32, %s_init: f32):",
            "      %one_f  = arith.constant 1.0 : f32",
            "      %mean   = arith.divf %sm, %nsz : f32",
            "      %msq    = arith.mulf %mean, %mean : f32",
            "      %esq    = arith.divf %sq, %nsz : f32",
            "      %var    = arith.subf %esq, %msq : f32",
            "      %vep    = arith.addf %var, %eps : f32",
            "      %std    = math.sqrt %vep : f32",
            "      %istd   = arith.divf %one_f, %std : f32",
            "      linalg.yield %mean, %istd : f32, f32",
        ]
        body += _lg("%mean_result, %istd_result", n_outer,
                    ["%sum_result : {}".format(tOuter), "%sumsq_result : {}".format(tOuter)],
                    ["%mean_dummy : {}".format(tOuter), "%istd_dummy : {}".format(tOuter)],
                    itypes_p2, maps_p2, body_p2)

    # Pass 3: normalize + affine
    # n_iters = nd; all parallel
    # input: identity, mean: outer, istd: outer, gamma: norm_dims, beta: norm_dims, out: identity
    norm_dims = list(range(batch_rank, nd))
    maps_p3 = [
        list(range(nd)),         # input
        list(range(batch_rank)), # mean
        list(range(batch_rank)), # istd
        norm_dims,               # gamma
        norm_dims,               # beta
        list(range(nd)),         # output
    ]
    if not outer_shape:
        maps_p3[1] = []; maps_p3[2] = []
    itypes_p3 = ["parallel"] * nd
    body_p3 = [
        "    ^bb0(%x: f32, %mn: f32, %is: f32, %g: f32, %b: f32, %init: f32):",
        "      %cent   = arith.subf %x, %mn : f32",
        "      %normed = arith.mulf %cent, %is : f32",
        "      %scaled = arith.mulf %normed, %g : f32",
        "      %res    = arith.addf %scaled, %b : f32",
        "      linalg.yield %res : f32",
    ]
    body += _lg("%result", nd,
                ["%input : {}".format(tA),
                 "%mean_result : {}".format(tOuter),
                 "%istd_result : {}".format(tOuter),
                 "%gamma : {}".format(tG),
                 "%beta : {}".format(tBt)],
                ["%out : {}".format(tA)],
                itypes_p3, maps_p3, body_p3)

    body.append("    return %result : {}".format(tA))
    return build_mlir(fname, [("input", inp), ("gamma", gamma_dims), ("beta", beta_dims)], body, inp)

# ---- conv2d ----

def gen_conv2d(stem, shapes):
    """2D convolution: NCHW input, FCHW filter via linalg.conv_2d_nchw_fchw.
    Adds tensor.pad for padded convolutions when input spatial dims are too small."""
    import math as _math
    if len(shapes) < 3:
        return None
    inp, filt, out = shapes[0], shapes[1], shapes[2]; fname = stem
    if len(inp) != 4 or len(filt) != 4:
        return None
    kH = filt[2] if isinstance(filt[2], int) else 3
    kW = filt[3] if isinstance(filt[3], int) else 3
    # Infer stride from spatial dims
    strides = 1
    if isinstance(inp[2], int) and isinstance(out[2], int) and out[2] > 0:
        strides = max(1, inp[2] // out[2])
    # Compute required padding based on convolution constraint:
    #   in_h: must be >= (out_h - 1)*s + d*(k-1) + 1
    dil = 1
    def _req_pad(in_sz, out_sz, k, s, d=1):
        if not (isinstance(in_sz, int) and isinstance(out_sz, int) and isinstance(k, int)):
            return 0
        req = (out_sz - 1) * s + d * (k - 1) + 1
        needed = req - in_sz
        return max(0, _math.ceil(needed / 2))
    pad_h = _req_pad(inp[2], out[2], kH, strides)
    pad_w = _req_pad(inp[3], out[3], kW, strides)

    tA = tensor_type(inp); tF = tensor_type(filt); tC = tensor_type(out)
    body = _consts(max(2, len(inp)))
    a_lines, a_d = _dims("input", inp)
    f_lines, f_d = _dims("filter", filt)
    body += a_lines + f_lines

    def _out_sz(i):
        d = out[i]
        if isinstance(d, int):
            v = "%out_sz{}".format(i)
            body.append("    {} = arith.constant {} : index".format(v, d))
            return v
        if i == 0: return a_d[0]
        if i == 1: return f_d[0]
        return a_d[i]

    out_dv = {}
    for i in range(4):
        if out[i] == "?":
            out_dv[i] = _out_sz(i)
    body.append(_empty(out, out_dv, name="%out"))

    # Add tensor.pad if needed
    if pad_h > 0 or pad_w > 0:
        inp_pad = list(inp)
        if isinstance(inp[2], int): inp_pad[2] = inp[2] + 2 * pad_h
        if isinstance(inp[3], int): inp_pad[3] = inp[3] + 2 * pad_w
        tA_pad = tensor_type(inp_pad)
        low_high = "[0, 0, {}, {}]".format(pad_h, pad_w)
        body.append("    %pad_cst = arith.constant 0.0 : f32")
        body.append("    %padded = tensor.pad %input low{} high{} {{".format(low_high, low_high))
        body.append("    ^bb0(%pi0: index, %pi1: index, %pi2: index, %pi3: index):")
        body.append("      tensor.yield %pad_cst : f32")
        body.append("    }} : {} to {}".format(tA, tA_pad))
        input_var = "%padded"
        tA_used = tA_pad
    else:
        input_var = "%input"
        tA_used = tA

    body.append("    %zero = arith.constant 0.0 : f32")
    body += _linalg_fill("%filled", "%zero", "%out", tC)
    body.append("    %result = linalg.conv_2d_nchw_fchw"
                " {{dilations = dense<1> : vector<2xi64>,"
                " strides = dense<{}> : vector<2xi64>}}".format(strides))
    body.append("      ins({}, %filter : {}, {})".format(input_var, tA_used, tF))
    body.append("      outs(%filled : {}) -> {}".format(tC, tC))
    body.append("    return %result : {}".format(tC))
    return build_mlir(fname, [("input", inp), ("filter", filt)], body, out)

# ---- max_pool2d ----

def gen_max_pool2d(stem, shapes):
    """2D max pool: NCHW layout via linalg.pooling_nchw_max."""
    if len(shapes) < 2:
        return None
    inp, out = shapes[0], shapes[1]
    if len(inp) < 4 or len(out) < 4:
        return None
    fname = stem
    strides = 2
    if isinstance(inp[2], int) and isinstance(out[2], int) and out[2] > 0:
        strides = max(1, inp[2] // out[2])
    kH = kW = strides

    tA = tensor_type(inp); tC = tensor_type(out)
    tKernel = "tensor<{}x{}xf32>".format(kH, kW)
    body = _consts(max(2, len(inp)))
    a_lines, a_d = _dims("input", inp)
    body += a_lines

    def _out_sz(i):
        d = out[i]
        if isinstance(d, int):
            v = "%po_sz{}".format(i)
            body.append("    {} = arith.constant {} : index".format(v, d))
            return v
        return a_d[i]

    out_dv = {}
    for i in range(4):
        if out[i] == "?":
            out_dv[i] = _out_sz(i)
    body.append(_empty(out, out_dv, name="%out"))
    body.append("    %neg_inf = arith.constant -3.4028234663852886e+38 : f32")
    body += _linalg_fill("%filled", "%neg_inf", "%out", tC)
    # Kernel dummy tensor (values irrelevant for max pooling)
    body.append("    %kernel = tensor.empty() : {}".format(tKernel))
    body.append("    %result = linalg.pooling_nchw_max"
                " {{dilations = dense<1> : vector<2xi64>,"
                " strides = dense<{}> : vector<2xi64>}}".format(strides))
    body.append("      ins(%input, %kernel : {}, {})".format(tA, tKernel))
    body.append("      outs(%filled : {}) -> {}".format(tC, tC))
    body.append("    return %result : {}".format(tC))
    return build_mlir(fname, [("input", inp)], body, out)

# ---- reduce_mean ----

def gen_reduce_mean(stem, shapes):
    """Reduce mean: linalg.generic reduction + divide."""
    if len(shapes) < 2:
        return None
    inp, out_declared = shapes[0], shapes[-1]; ndA = len(inp)
    reduce_dims, extra_out = find_reduce_dims(inp, out_declared)
    if not reduce_dims:
        reduce_dims = [ndA - 1]; extra_out = []
    kept = [i for i in range(ndA) if i not in set(reduce_dims)]
    out_mid = [inp[i] for i in kept]

    fname = stem
    tInp = tensor_type(inp)
    tMid = tensor_type(out_mid) if out_mid else "tensor<f32>"

    body = _consts(max(2, ndA))
    a_lines, a_d = _dims("input", inp)
    body += a_lines

    body.append(_empty(out_mid, {i: a_d[kept[i]] for i in range(len(out_mid)) if out_mid[i] == "?"}, name="%sum_t")
                if out_mid else "    %sum_t = tensor.empty() : tensor<f32>")
    body.append("    %zero_f = arith.constant 0.0 : f32")
    body += _linalg_fill("%sum_init", "%zero_f", "%sum_t", tMid)

    # Build index maps
    n_iters = ndA
    in_map  = list(range(ndA))
    out_map = list(kept)
    iter_types = ["parallel" if i in kept else "reduction" for i in range(ndA)]

    region = [
        "    ^bb0(%in: f32, %acc: f32):",
        "      %ns = arith.addf %acc, %in : f32",
        "      linalg.yield %ns : f32",
    ]
    body += _lg("%sum_result", n_iters,
                ["%input : {}".format(tInp)],
                ["%sum_init : {}".format(tMid)],
                iter_types, [in_map, out_map], region)

    # Compute reciprocal scale
    red_size = 1
    dyn_red = False
    for d in reduce_dims:
        if isinstance(inp[d], int): red_size *= inp[d]
        else: dyn_red = True; break
    if dyn_red:
        body.append("    %rsz_idx_0 = arith.constant 1 : index")
        for ii, d in enumerate(reduce_dims):
            body.append("    %rsz_idx_{} = arith.muli %rsz_idx_{}, {} : index".format(
                ii + 1, ii, a_d[d]))
        body.append("    %rsz_i64 = arith.index_cast %rsz_idx_{} : index to i64".format(len(reduce_dims)))
        body.append("    %rsz    = arith.sitofp %rsz_i64 : i64 to f32")
        body.append("    %scale  = arith.constant 1.0 : f32")  # will divide, not multiply
    else:
        body.append("    %scale = arith.constant {} : f32".format(1.0 / red_size))

    # Divide result
    body.append(_empty(out_mid, {i: a_d[kept[i]] for i in range(len(out_mid)) if out_mid[i] == "?"}, name="%div_t")
                if out_mid else "    %div_t = tensor.empty() : tensor<f32>")
    body += _linalg_fill("%div_init", "%zero_f", "%div_t", tMid)

    n_mid = len(out_mid)
    maps_div = [list(range(n_mid)), list(range(n_mid))]
    itypes_div = ["parallel"] * n_mid if n_mid > 0 else ["parallel"]
    if dyn_red:
        region_div = [
            "    ^bb0(%sv: f32, %init: f32):",
            "      %res = arith.divf %sv, %rsz : f32",
            "      linalg.yield %res : f32",
        ]
    else:
        region_div = [
            "    ^bb0(%sv: f32, %init: f32):",
            "      %res = arith.mulf %sv, %scale : f32",
            "      linalg.yield %res : f32",
        ]

    if n_mid > 0:
        body += _lg("%mean_result", n_mid,
                    ["%sum_result : {}".format(tMid)],
                    ["%div_init : {}".format(tMid)],
                    itypes_div, maps_div, region_div)
        out_res = "%mean_result"
    else:
        # Scalar case
        body += [
            "    %c0_rm = arith.constant 0 : index",
            "    %sv0   = tensor.extract %sum_result[] : tensor<f32>",
            "    %res0  = arith.mulf %sv0, %scale : f32",
            "    %mean_result = tensor.insert %res0 into %div_init[] : tensor<f32>",
        ]
        out_res = "%mean_result"

    if extra_out:
        nextra = len(extra_out)
        if len(out_mid) > 0:
            reassoc = [[i] for i in range(len(out_mid) - 1)] + [list(range(len(out_mid) - 1, len(out_mid) + nextra))]
        else:
            reassoc = [list(range(nextra))]
        reassoc_str = "[" + ", ".join("[" + ", ".join(str(x) for x in g) + "]" for g in reassoc) + "]"
        tFinal = tensor_type(out_declared)
        out_shape_str = "[" + ", ".join(str(d) for d in out_declared) + "]"
        body += [
            "    %result = tensor.expand_shape {} {} output_shape {} : {} into {}".format(
                out_res, reassoc_str, out_shape_str, tMid, tFinal),
            "    return %result : {}".format(tFinal),
        ]
        return build_mlir(fname, [("input", inp)], body, out_declared)
    else:
        body.append("    return {} : {}".format(out_res, tMid))
        return build_mlir(fname, [("input", inp)], body, out_mid)

# ---- embedding ----

def gen_embedding(stem, shapes):
    """Embedding lookup: output[b,s,d] = table[indices[b,s], d]."""
    if len(shapes) < 3:
        return None
    idx_dims, tbl_dims, out_dims = shapes[0], shapes[1], shapes[2]
    nd_idx = len(idx_dims); nd_tbl = len(tbl_dims); nd_out = len(out_dims); fname = stem
    tIdx = tensor_type(idx_dims, "i64"); tTbl = tensor_type(tbl_dims)
    tOut = tensor_type(out_dims)
    body = _consts(max(2, nd_out))
    a_lines, a_d = _dims("indices", idx_dims, dtype="i64")
    t_lines, t_d = _dims("table",   tbl_dims)
    body += a_lines + t_lines
    out_dv = {}
    for i in range(nd_idx):
        if out_dims[i] == "?": out_dv[i] = a_d[i]
    for ei in range(nd_tbl - 1):
        oi = nd_idx + ei
        if oi < nd_out and out_dims[oi] == "?": out_dv[oi] = t_d[ei + 1]
    body.append(_empty(out_dims, out_dv, name="%out"))
    body.append("    %zero_emb = arith.constant 0.0 : f32")
    body += _linalg_fill("%out_init", "%zero_emb", "%out", tOut)

    # n_iters = nd_out (all parallel)
    # indices maps to first nd_idx dims
    n_out = nd_out
    idx_map  = list(range(nd_idx))
    out_map  = list(range(nd_out))
    # Capture %table in the region via tensor.extract + linalg.index
    emb_dims = list(range(nd_tbl - 1))  # embed dimension indices within nd_out
    region = [
        "    ^bb0(%raw_idx: i64, %init: f32):",
    ]
    for ei in range(nd_tbl - 1):
        region.append("      %ei{} = linalg.index {} : index".format(ei, nd_idx + ei))
    region += [
        "      %row = arith.index_cast %raw_idx : i64 to index",
        "      %val = tensor.extract %table[%row{}] : {}".format(
            "".join(", %ei{}".format(ei) for ei in range(nd_tbl - 1)), tTbl),
        "      linalg.yield %val : f32",
    ]
    body += _lg("%result", n_out,
                ["%indices : {}".format(tIdx)],
                ["%out_init : {}".format(tOut)],
                ["parallel"] * n_out, [idx_map, out_map], region)
    body.append("    return %result : {}".format(tOut))
    return build_mlir(fname, [("indices", idx_dims, "i64"), ("table", tbl_dims)], body, out_dims)

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

GENERATORS = {
    "matmul":              gen_matmul,
    "elemwise_add":        gen_elemwise_add,
    "relu":                gen_relu,
    "sigmoid":             gen_sigmoid,
    "gelu":                gen_gelu,
    "softmax":             gen_softmax,
    "transpose":           gen_transpose,
    "reshape":             gen_reshape,
    "concat":              gen_concat,
    "batch_norm":          gen_batch_norm,
    "layer_normalization": gen_layer_norm,
    "conv2d":              gen_conv2d,
    "max_pool2d":          gen_max_pool2d,
    "reduce_mean":         gen_reduce_mean,
    "embedding":           gen_embedding,
}

def generate_for_file(co_path, category, dry_run=False, verbose=False):
    stem      = co_path.stem
    shapes    = shapes_for_category(stem, category)
    gen_fn    = GENERATORS.get(category)
    if gen_fn is None:
        if verbose: print("  [SKIP] No generator for category {}".format(category))
        return False
    mlir_code = gen_fn(stem, shapes)
    if mlir_code is None:
        if verbose: print("  [SKIP] Generator returned None for {}".format(stem))
        return False
    out_path = IREE_DIR / category / (stem + ".mlir")
    if verbose: print("  -> {}".format(out_path.relative_to(ROOT)))
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(mlir_code)
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Generate IREE linalg-on-tensors MLIR benchmark cases")
    parser.add_argument("--dry-run",  action="store_true", help="Print counts, don't write files")
    parser.add_argument("--category", default=None, help="Only generate this category")
    parser.add_argument("--verbose",  action="store_true", help="Print each output path")
    args = parser.parse_args()

    categories = sorted(c.name for c in CHOREO_DIR.iterdir() if c.is_dir())
    if args.category:
        categories = [args.category]

    total, ok = 0, 0
    for cat in categories:
        co_files = sorted((CHOREO_DIR / cat).glob("*.co"))
        if not co_files:
            continue
        if args.verbose:
            print("\n=== {} ({} files) ===".format(cat, len(co_files)))
        for cf in co_files:
            total += 1
            if generate_for_file(cf, cat, dry_run=args.dry_run, verbose=args.verbose):
                ok += 1
            elif args.verbose:
                pass

    print("Generated {}/{} IREE MLIR cases.".format(ok, total))
    if ok < total:
        print("  {} cases skipped (unsupported shapes or generators returned None).".format(total - ok))

if __name__ == "__main__":
    main()
