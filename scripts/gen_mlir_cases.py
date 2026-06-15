#!/usr/bin/env python3
"""Generate MLIR benchmarks: tensor<> + scf.for + tensor.extract/insert + cf.assert.

Each function:
  - Uses tensor<> types (value semantics, no linalg)
  - Reads with tensor.extract, writes with tensor.insert (carrying tensor through iter_args)
  - Declares shape constraints via cf.assert + arith.cmpi eq
"""

import re
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
CHOREO_DIR = ROOT / "benchmark/choreo"
MLIR_DIR   = ROOT / "benchmark/mlir/cases"

# ---------------------------------------------------------------------------
# Shape utilities
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

def _assert_eq(v1, v2, msg, cnt):
    eq = "%eq{}".format(cnt)
    return ([
        "    {} = arith.cmpi eq, {}, {} : index".format(eq, v1, v2),
        "    cf.assert {}, \"{}\"".format(eq, msg),
    ], cnt + 1)

# ---------------------------------------------------------------------------
# Loop helpers (tensor iter_args style)
# ---------------------------------------------------------------------------

def _nested_for_tensor(specs, body_inner, carry_var, carry_type):
    """
    Build nested scf.for loops where the outermost / innermost carry a tensor.
    specs: [(loopvar, hi), ...] outermost first.
    The innermost loop body (body_inner) must yield carry_type.
    Each outer loop carries the updated tensor as iter_args.
    Returns list of lines.
    """
    if not specs:
        return list(body_inner)
    var, hi = specs[0]
    if len(specs) == 1:
        inner = list(body_inner)
    else:
        inner_carry = "%tc_{}".format(var.lstrip("%"))
        inner_lines = _nested_for_tensor(specs[1:], body_inner, inner_carry, carry_type)
        inner = inner_lines

    lines = [
        "    %tc_{} = scf.for {} = %c0 to {} step %c1 iter_args(%tc_{}_arg = {}) -> ({}) {{".format(
            var.lstrip("%"), var, hi, var.lstrip("%"), carry_var, carry_type)
    ]
    # Remap carry_var usage inside to the iter_arg name
    # The inner code references carry_var; we need to substitute with the iter_arg
    # We do this by passing the iter_arg name down instead
    return _nested_for_tensor_impl(specs, body_inner, carry_var, carry_type)

def _nested_for_tensor_impl(specs, innermost_body, init_carry, carry_type):
    """
    Build nested scf.for with correct iter_args chaining.
    specs: [(loopvar, hi), ...] outermost first.
    innermost_body: lines inside the innermost loop body (ending with scf.yield %carry).
    Returns flat list of lines for the outermost for, ending with %result_var = scf.for ...
    The outermost result is accessible as %t_res_{outermost_var}.
    """
    def build(depth, carry):
        var, hi = specs[depth]
        res_name = "%t_res_{}".format(var.lstrip("%"))
        arg_name = "%t_arg_{}".format(var.lstrip("%"))
        header = "    {} = scf.for {} = %c0 to {} step %c1 iter_args({} = {}) -> ({}) {{".format(
            res_name, var, hi, arg_name, carry, carry_type)
        if depth == len(specs) - 1:
            # innermost: substitute %CARRY in body with arg_name
            body = [l.replace("%%CARRY%%", arg_name) for l in innermost_body]
            return [header] + body + ["    }"]
        else:
            inner = build(depth + 1, arg_name)
            # The inner result is used to yield
            inner_res = "%t_res_{}".format(specs[depth + 1][0].lstrip("%"))
            return [header] + inner + ["    scf.yield {} : {}".format(inner_res, carry_type), "    }"]
    return build(0, init_carry)

# ---------------------------------------------------------------------------
# Filename / shape parsing helpers
# ---------------------------------------------------------------------------

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
# Category generators
# ---------------------------------------------------------------------------

# ---- matmul ----

def gen_matmul(stem, shapes):
    if len(shapes) < 3: return None
    A, B, C = shapes[0], shapes[1], shapes[2]
    fname = stem; ndA, ndB = len(A), len(B)
    nd_max = max(ndA, ndB, len(C))
    body = _consts(max(2, nd_max))
    body.append("    %zero_f = arith.constant 0.0 : f32")
    a_lines, a_d = _dims("lhs", A); b_lines, b_d = _dims("rhs", B)
    body += a_lines + b_lines
    eq_cnt = 0
    tA = tensor_type(A); tB = tensor_type(B); tC = tensor_type(C)

    def _dyn_sz(d, i, fallback_dv):
        if isinstance(d, int):
            v = "%csz_{}_{}".format(i, d)
            body.append("    {} = arith.constant {} : index".format(v, d))
            return v
        return fallback_dv

    if ndA == 2 and ndB == 2:
        lines, eq_cnt = _assert_eq(a_d[1], b_d[0], "lhs.dim(1)==rhs.dim(0)", eq_cnt); body += lines
        out_dv = {0: a_d[0], 1: b_d[1]}
        body.append(_empty(C, out_dv))
        # i-loop carries tensor
        body.append("    %t_res_i = scf.for %i = %c0 to {} step %c1 iter_args(%t_arg_i = %out) -> ({}) {{".format(a_d[0], tC))
        body.append("    %t_res_j = scf.for %j = %c0 to {} step %c1 iter_args(%t_arg_j = %t_arg_i) -> ({}) {{".format(b_d[1], tC))
        body.append("    %acc = scf.for %k = %c0 to {} step %c1 iter_args(%s = %zero_f) -> (f32) {{".format(a_d[1]))
        body += [
            "    %a   = tensor.extract %lhs[%i, %k] : {}".format(tA),
            "    %b   = tensor.extract %rhs[%k, %j] : {}".format(tB),
            "    %p   = arith.mulf %a, %b : f32",
            "    %ns  = arith.addf %s, %p : f32",
            "    scf.yield %ns : f32",
            "    }",
            "    %t_new = tensor.insert %acc into %t_arg_j[%i, %j] : {}".format(tC),
            "    scf.yield %t_new : {}".format(tC),
            "    }",
            "    scf.yield %t_res_j : {}".format(tC),
            "    }",
        ]

    elif ndA == 3 and ndB == 3:
        lines, eq_cnt = _assert_eq(a_d[0], b_d[0], "lhs.dim(0)==rhs.dim(0)", eq_cnt); body += lines
        lines, eq_cnt = _assert_eq(a_d[2], b_d[1], "lhs.dim(2)==rhs.dim(1)", eq_cnt); body += lines
        out_dv = {0: a_d[0], 1: a_d[1], 2: b_d[2]}
        body.append(_empty(C, out_dv))
        body.append("    %t_res_bs = scf.for %bs = %c0 to {} step %c1 iter_args(%t_arg_bs = %out) -> ({}) {{".format(a_d[0], tC))
        body.append("    %t_res_i = scf.for %i = %c0 to {} step %c1 iter_args(%t_arg_i = %t_arg_bs) -> ({}) {{".format(a_d[1], tC))
        body.append("    %t_res_j = scf.for %j = %c0 to {} step %c1 iter_args(%t_arg_j = %t_arg_i) -> ({}) {{".format(b_d[2], tC))
        body.append("    %acc = scf.for %k = %c0 to {} step %c1 iter_args(%s = %zero_f) -> (f32) {{".format(a_d[2]))
        body += [
            "    %a   = tensor.extract %lhs[%bs, %i, %k] : {}".format(tA),
            "    %b   = tensor.extract %rhs[%bs, %k, %j] : {}".format(tB),
            "    %p   = arith.mulf %a, %b : f32",
            "    %ns  = arith.addf %s, %p : f32",
            "    scf.yield %ns : f32",
            "    }",
            "    %t_new = tensor.insert %acc into %t_arg_j[%bs, %i, %j] : {}".format(tC),
            "    scf.yield %t_new : {}".format(tC),
            "    }",
            "    scf.yield %t_res_j : {}".format(tC),
            "    }",
            "    scf.yield %t_res_i : {}".format(tC),
            "    }",
        ]

    elif ndA == 3 and ndB == 2:
        lines, eq_cnt = _assert_eq(a_d[2], b_d[0], "lhs.dim(2)==rhs.dim(0)", eq_cnt); body += lines
        out_dv = {0: a_d[0], 1: a_d[1], 2: b_d[1]}
        body.append(_empty(C, out_dv))
        body.append("    %t_res_bs = scf.for %bs = %c0 to {} step %c1 iter_args(%t_arg_bs = %out) -> ({}) {{".format(a_d[0], tC))
        body.append("    %t_res_i = scf.for %i = %c0 to {} step %c1 iter_args(%t_arg_i = %t_arg_bs) -> ({}) {{".format(a_d[1], tC))
        body.append("    %t_res_j = scf.for %j = %c0 to {} step %c1 iter_args(%t_arg_j = %t_arg_i) -> ({}) {{".format(b_d[1], tC))
        body.append("    %acc = scf.for %k = %c0 to {} step %c1 iter_args(%s = %zero_f) -> (f32) {{".format(a_d[2]))
        body += [
            "    %a   = tensor.extract %lhs[%bs, %i, %k] : {}".format(tA),
            "    %b   = tensor.extract %rhs[%k, %j] : {}".format(tB),
            "    %p   = arith.mulf %a, %b : f32",
            "    %ns  = arith.addf %s, %p : f32",
            "    scf.yield %ns : f32",
            "    }",
            "    %t_new = tensor.insert %acc into %t_arg_j[%bs, %i, %j] : {}".format(tC),
            "    scf.yield %t_new : {}".format(tC),
            "    }",
            "    scf.yield %t_res_j : {}".format(tC),
            "    }",
            "    scf.yield %t_res_i : {}".format(tC),
            "    }",
        ]

    elif ndA == 4 and ndB == 4:
        lines, eq_cnt = _assert_eq(a_d[0], b_d[0], "lhs.dim(0)==rhs.dim(0)", eq_cnt); body += lines
        lines, eq_cnt = _assert_eq(a_d[1], b_d[1], "lhs.dim(1)==rhs.dim(1)", eq_cnt); body += lines
        lines, eq_cnt = _assert_eq(a_d[3], b_d[2], "lhs.dim(3)==rhs.dim(2)", eq_cnt); body += lines
        out_dv = {0: a_d[0], 1: a_d[1], 2: a_d[2], 3: b_d[3]}
        body.append(_empty(C, out_dv))
        body.append("    %t_res_b0 = scf.for %b0 = %c0 to {} step %c1 iter_args(%t_arg_b0 = %out) -> ({}) {{".format(a_d[0], tC))
        body.append("    %t_res_b1 = scf.for %b1 = %c0 to {} step %c1 iter_args(%t_arg_b1 = %t_arg_b0) -> ({}) {{".format(a_d[1], tC))
        body.append("    %t_res_i = scf.for %i = %c0 to {} step %c1 iter_args(%t_arg_i = %t_arg_b1) -> ({}) {{".format(a_d[2], tC))
        body.append("    %t_res_j = scf.for %j = %c0 to {} step %c1 iter_args(%t_arg_j = %t_arg_i) -> ({}) {{".format(b_d[3], tC))
        body.append("    %acc = scf.for %k = %c0 to {} step %c1 iter_args(%s = %zero_f) -> (f32) {{".format(a_d[3]))
        body += [
            "    %a   = tensor.extract %lhs[%b0, %b1, %i, %k] : {}".format(tA),
            "    %b   = tensor.extract %rhs[%b0, %b1, %k, %j] : {}".format(tB),
            "    %p   = arith.mulf %a, %b : f32",
            "    %ns  = arith.addf %s, %p : f32",
            "    scf.yield %ns : f32",
            "    }",
            "    %t_new = tensor.insert %acc into %t_arg_j[%b0, %b1, %i, %j] : {}".format(tC),
            "    scf.yield %t_new : {}".format(tC),
            "    }",
            "    scf.yield %t_res_j : {}".format(tC),
            "    }",
            "    scf.yield %t_res_i : {}".format(tC),
            "    }",
            "    scf.yield %t_res_b1 : {}".format(tC),
            "    }",
        ]
    else:
        return None

    _ret_map = {2: "i", 3: "bs", 4: "b0"}
    body.append("    return %t_res_{} : {}".format(_ret_map.get(ndA, "i"), tC))
    return build_mlir(fname, [("lhs", A), ("rhs", B)], body, C)

# ---- elemwise_add ----

def gen_elemwise_add(stem, shapes):
    if len(shapes) < 3: return None
    A, B, C = shapes[0], shapes[1], shapes[2]
    fname = stem; ndA, ndB, ndC = len(A), len(B), len(C)
    body = _consts(max(2, ndA, ndB, ndC))
    a_lines, a_d = _dims("lhs", A); b_lines, b_d = _dims("rhs", B)
    body += a_lines + b_lines
    eq_cnt = 0; tA = tensor_type(A); tB = tensor_type(B); tC = tensor_type(C)
    lvars = ["%ei{}".format(i) for i in range(ndC)]
    idxs  = ", ".join(lvars)

    if ndA == ndB:
        for i in range(ndA):
            lines, eq_cnt = _assert_eq(a_d[i], b_d[i], "lhs.dim({})==rhs.dim({})".format(i, i), eq_cnt)
            body += lines
        body.append(_empty(C, {i: a_d[i] for i in range(ndC)}))
    elif ndB == 1:
        body.append(_empty(C, {i: a_d[i] for i in range(ndC)}))
        if B[0] != 1:
            b_dim_idx = ndC - 1
            for j, cv in enumerate(C):
                if cv == B[0]: b_dim_idx = j; break
            lines, eq_cnt = _assert_eq(b_d[0], a_d[b_dim_idx],
                "rhs.dim(0)==lhs.dim({})".format(b_dim_idx), eq_cnt)
            body += lines
    else:
        body.append(_empty(C, {i: a_d[i] for i in range(ndC)}))

    # Build nested iter_args loops
    carry = "%out"; t = tC
    for depth in range(ndC):
        v = lvars[depth]; hi = a_d[depth] if depth < ndA else a_d[0]
        res = "%t_res_{}".format(v.lstrip("%"))
        body.append("    {} = scf.for {} = %c0 to {} step %c1 iter_args(%t_ea_{} = {}) -> ({}) {{".format(
            res, v, hi, v.lstrip("%"), carry, t))
        carry = "%t_ea_{}".format(v.lstrip("%"))

    body.append("    %va = tensor.extract %lhs[{}] : {}".format(idxs, tA))
    if ndA == ndB:
        body.append("    %vb = tensor.extract %rhs[{}] : {}".format(idxs, tB))
    elif ndB == 1 and B[0] == 1:
        body.append("    %zidx = arith.constant 0 : index")
        body.append("    %vb = tensor.extract %rhs[%zidx] : {}".format(tB))
    elif ndB == 1:
        b_dim_idx = ndC - 1
        for j, cv in enumerate(C):
            if cv == B[0]: b_dim_idx = j; break
        body.append("    %vb = tensor.extract %rhs[{}] : {}".format(lvars[b_dim_idx], tB))
    else:
        body.append("    %va2 = tensor.extract %lhs[{}] : {}".format(idxs, tA))
        body.append("    %vb = %va2")
    body.append("    %rs = arith.addf %va, %vb : f32")
    body.append("    %t_ins = tensor.insert %rs into {}[{}] : {}".format(carry, idxs, tC))
    body.append("    scf.yield %t_ins : {}".format(tC))

    for depth in range(ndC - 1, -1, -1):
        v = lvars[depth]; res = "%t_res_{}".format(v.lstrip("%"))
        body.append("    }")
        if depth > 0:
            p = lvars[depth - 1]; pres = "%t_res_{}".format(v.lstrip("%"))
            body.append("    scf.yield {} : {}".format(res, tC))

    body.append("    return {} : {}".format(
        "%t_res_{}".format(lvars[0].lstrip("%")), tC))
    return build_mlir(fname, [("lhs", A), ("rhs", B)], body, C)

# ---- unary pointwise ----

def _gen_unary(stem, shapes, compute_lines):
    if not shapes: return None
    A = shapes[0]; nd = len(A); fname = stem; tA = tensor_type(A)
    body = _consts(max(2, nd))
    a_lines, a_d = _dims("input", A); body += a_lines
    body.append(_empty(A, {i: a_d[i] for i in range(nd) if A[i] == "?"}))
    lvars = ["%ui{}".format(i) for i in range(nd)]
    idxs  = ", ".join(lvars)
    carry = "%out"; t = tA
    for depth in range(nd):
        v = lvars[depth]; hi = a_d[depth]
        res = "%t_res_{}".format(v.lstrip("%"))
        body.append("    {} = scf.for {} = %c0 to {} step %c1 iter_args(%t_un_{} = {}) -> ({}) {{".format(
            res, v, hi, v.lstrip("%"), carry, t))
        carry = "%t_un_{}".format(v.lstrip("%"))
    body.append("    %in_val = tensor.extract %input[{}] : {}".format(idxs, tA))
    body += compute_lines
    body.append("    %t_ins = tensor.insert %out_val into {}[{}] : {}".format(carry, idxs, tA))
    body.append("    scf.yield %t_ins : {}".format(tA))
    for depth in range(nd - 1, -1, -1):
        v = lvars[depth]; res = "%t_res_{}".format(v.lstrip("%"))
        body.append("    }")
        if depth > 0:
            body.append("    scf.yield {} : {}".format(res, tA))
    body.append("    return {} : {}".format("%t_res_{}".format(lvars[0].lstrip("%")), tA))
    return build_mlir(fname, [("input", A)], body, A)

def gen_relu(stem, shapes):
    return _gen_unary(stem, shapes, [
        "    %zf = arith.constant 0.0 : f32",
        "    %out_val = arith.maximumf %in_val, %zf : f32",
    ])

def gen_sigmoid(stem, shapes):
    return _gen_unary(stem, shapes, [
        "    %neg = arith.negf %in_val : f32",
        "    %exp = math.exp %neg : f32",
        "    %one = arith.constant 1.0 : f32",
        "    %den = arith.addf %one, %exp : f32",
        "    %out_val = arith.divf %one, %den : f32",
    ])

def gen_gelu(stem, shapes):
    return _gen_unary(stem, shapes, [
        "    %half   = arith.constant 0.5 : f32",
        "    %isqrt2 = arith.constant 0.7071067811865476 : f32",
        "    %sc     = arith.mulf %in_val, %isqrt2 : f32",
        "    %efv    = math.erf %sc : f32",
        "    %one    = arith.constant 1.0 : f32",
        "    %ep1    = arith.addf %efv, %one : f32",
        "    %hx     = arith.mulf %in_val, %half : f32",
        "    %out_val = arith.mulf %hx, %ep1 : f32",
    ])

# ---- softmax ----

def gen_softmax(stem, shapes):
    if not shapes: return None
    A = shapes[0]; nd = len(A); fname = stem; tA = tensor_type(A)
    body = _consts(max(2, nd))
    a_lines, a_d = _dims("input", A); body += a_lines
    body.append(_empty(A, {i: a_d[i] for i in range(nd) if A[i] == "?"}))
    last_hi     = a_d[nd - 1]
    outer_lvars = ["%so{}".format(i) for i in range(nd - 1)]
    sk = "%sk"

    def fidx(k):
        return "{}, {}".format(", ".join(outer_lvars), k) if outer_lvars else k

    # Build outer iter_args loops for the output tensor
    carry = "%out"; t = tA
    for depth in range(nd - 1):
        v = outer_lvars[depth]; hi = a_d[depth]
        res = "%t_res_{}".format(v.lstrip("%"))
        body.append("    {} = scf.for {} = %c0 to {} step %c1 iter_args(%t_sm_{} = {}) -> ({}) {{".format(
            res, v, hi, v.lstrip("%"), carry, t))
        carry = "%t_sm_{}".format(v.lstrip("%"))

    # Pass 1: max (scalar iter_args)
    body += [
        "    %neg_inf = arith.constant -3.4028234663852886e+38 : f32",
        "    %max_val = scf.for {} = %c0 to {} step %c1 iter_args(%mx = %neg_inf) -> (f32) {{".format(sk, last_hi),
        "    %pv1 = tensor.extract %input[{}] : {}".format(fidx(sk), tA),
        "    %gt  = arith.cmpf ogt, %pv1, %mx : f32",
        "    %nx  = arith.select %gt, %pv1, %mx : f32",
        "    scf.yield %nx : f32",
        "    }",
        # Pass 2: sum of exp (scalar iter_args)
        "    %zero_f  = arith.constant 0.0 : f32",
        "    %sum_val = scf.for {} = %c0 to {} step %c1 iter_args(%sm = %zero_f) -> (f32) {{".format(sk, last_hi),
        "    %pv2     = tensor.extract %input[{}] : {}".format(fidx(sk), tA),
        "    %shifted = arith.subf %pv2, %max_val : f32",
        "    %expv    = math.exp %shifted : f32",
        "    %ns2     = arith.addf %sm, %expv : f32",
        "    scf.yield %ns2 : f32",
        "    }",
        # Pass 3: write via tensor iter_args
        "    %t_res_{} = scf.for {} = %c0 to {} step %c1 iter_args(%t_sm_sk = {}) -> ({}) {{".format(
            sk.lstrip("%"), sk, last_hi, carry, t),
        "    %pv3  = tensor.extract %input[{}] : {}".format(fidx(sk), tA),
        "    %sh3  = arith.subf %pv3, %max_val : f32",
        "    %ex3  = math.exp %sh3 : f32",
        "    %norm = arith.divf %ex3, %sum_val : f32",
        "    %t_sk_ins = tensor.insert %norm into %t_sm_sk[{}] : {}".format(fidx(sk), tA),
        "    scf.yield %t_sk_ins : {}".format(tA),
        "    }",
    ]

    last_res = "%t_res_sk"
    for depth in range(nd - 2, -1, -1):
        v = outer_lvars[depth]; res = "%t_res_{}".format(v.lstrip("%"))
        body.append("    scf.yield {} : {}".format(last_res, tA))
        last_res = res
        body.append("    }")

    body.append("    return {} : {}".format(last_res, tA))
    return build_mlir(fname, [("input", A)], body, A)

# ---- transpose ----

def gen_transpose(stem, shapes):
    if len(shapes) < 2: return None
    A, C = shapes[0], shapes[1]; nd = len(A); fname = stem
    tA = tensor_type(A); tC = tensor_type(C)
    used = [False] * nd; perm = []
    for cv in C:
        best = None
        for ai, av in enumerate(A):
            if not used[ai] and av == cv: best = ai; break
        if best is None:
            for ai in range(nd):
                if not used[ai]: best = ai; break
        perm.append(best); used[best] = True
    perm_inv = [0] * nd
    for oi, ai in enumerate(perm): perm_inv[ai] = oi
    body = _consts(max(2, nd))
    a_lines, a_d = _dims("input", A); body += a_lines
    out_dv = {i: a_d[perm[i]] for i in range(nd)}
    body.append(_empty(C, out_dv))
    lvars   = ["%pi{}".format(i) for i in range(nd)]
    out_idx = ", ".join(lvars)
    in_idx  = ", ".join(lvars[perm_inv[i]] for i in range(nd))
    carry = "%out"; t = tC
    for depth in range(nd):
        v = lvars[depth]; hi = a_d[perm[depth]]
        res = "%t_res_{}".format(v.lstrip("%"))
        body.append("    {} = scf.for {} = %c0 to {} step %c1 iter_args(%t_tr_{} = {}) -> ({}) {{".format(
            res, v, hi, v.lstrip("%"), carry, t))
        carry = "%t_tr_{}".format(v.lstrip("%"))
    body.append("    %tv = tensor.extract %input[{}] : {}".format(in_idx, tA))
    body.append("    %t_ins = tensor.insert %tv into {}[{}] : {}".format(carry, out_idx, tC))
    body.append("    scf.yield %t_ins : {}".format(tC))
    for depth in range(nd - 1, -1, -1):
        v = lvars[depth]; res = "%t_res_{}".format(v.lstrip("%"))
        body.append("    }")
        if depth > 0:
            body.append("    scf.yield {} : {}".format(res, tC))
    body.append("    return {} : {}".format("%t_res_{}".format(lvars[0].lstrip("%")), tC))
    return build_mlir(fname, [("input", A)], body, C)

# ---- reshape ----

def _reshape_placeholder(fname, A, C):
    ndA = len(A); ndC = len(C); tA = tensor_type(A); tC = tensor_type(C)
    body = _consts(max(2, ndA))
    a_lines, a_d = _dims("input", A); body += a_lines
    in_dyn = [a_d[i] for i in range(ndA) if A[i] == "?"]
    out_dv = {}
    for i, d in enumerate(C):
        if d == "?":
            out_dv[i] = in_dyn.pop(0) if in_dyn else a_d[0]
    body.append(_empty(C, out_dv))
    body.append("    %zf = arith.constant 0.0 : f32")
    sz_vars = []
    for i, d in enumerate(C):
        if isinstance(d, int):
            sv = "%rsz{}".format(i); body.append("    {} = arith.constant {} : index".format(sv, d)); sz_vars.append(sv)
        else: sz_vars.append(out_dv[i])
    lvars = ["%rp{}".format(i) for i in range(ndC)]; idxs = ", ".join(lvars)
    carry = "%out"; t = tC
    for depth in range(ndC):
        v = lvars[depth]; hi = sz_vars[depth]
        res = "%t_res_{}".format(v.lstrip("%"))
        body.append("    {} = scf.for {} = %c0 to {} step %c1 iter_args(%t_ph_{} = {}) -> ({}) {{".format(
            res, v, hi, v.lstrip("%"), carry, t))
        carry = "%t_ph_{}".format(v.lstrip("%"))
    body.append("    %t_ins = tensor.insert %zf into {}[{}] : {}".format(carry, idxs, tC))
    body.append("    scf.yield %t_ins : {}".format(tC))
    for depth in range(ndC - 1, -1, -1):
        v = lvars[depth]; res = "%t_res_{}".format(v.lstrip("%"))
        body.append("    }")
        if depth > 0:
            body.append("    scf.yield {} : {}".format(res, tC))
    body.append("    return {} : {}".format("%t_res_{}".format(lvars[0].lstrip("%")), tC))
    return build_mlir(fname, [("input", A)], body, C)

def _gen_reshape_expand(fname, A, C):
    reassoc = _find_expand_reassoc(A, C)
    if reassoc is None: return _reshape_placeholder(fname, A, C)
    reassoc_str = "[" + ", ".join("[" + ", ".join(str(x) for x in g) + "]" for g in reassoc) + "]"
    tA = tensor_type(A); tC = tensor_type(C)
    # Build output_shape list
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
            return _reshape_placeholder(fname, A, C)
    out_shape_str = "[" + ", ".join(out_tokens) + "]"
    body = preamble + [
        "    %out = tensor.expand_shape %input {} output_shape {} : {} into {}".format(
            reassoc_str, out_shape_str, tA, tC),
        "    return %out : {}".format(tC),
    ]
    return build_mlir(fname, [("input", A)], body, C)

def _gen_reshape_collapse(fname, A, C):
    ndA, ndC = len(A), len(C)
    reassoc = [[] for _ in range(ndC)]
    ci = 0
    for ai in range(ndA):
        if ci < ndC - 1 and A[ai] == C[ci]: reassoc[ci].append(ai); ci += 1
        else: reassoc[ci].append(ai)
    for i in range(ndC):
        if not reassoc[i]:
            if i > 0 and len(reassoc[i - 1]) > 1: reassoc[i].append(reassoc[i - 1].pop())
    for i, grp in enumerate(reassoc):
        grp_dims = [A[j] for j in grp]
        has_dyn = any(d == "?" for d in grp_dims); out_d = C[i]
        if has_dyn and isinstance(out_d, int): return _reshape_placeholder(fname, A, C)
        if not has_dyn:
            prod = 1
            for d in grp_dims: prod *= d
            if isinstance(out_d, int) and prod != out_d: return _reshape_placeholder(fname, A, C)
    reassoc_str = "[" + ", ".join("[" + ", ".join(str(x) for x in g) + "]" for g in reassoc) + "]"
    tA = tensor_type(A); tC = tensor_type(C)
    lines = [
        "    %out = tensor.collapse_shape %input {} : {} into {}".format(reassoc_str, tA, tC),
        "    return %out : {}".format(tC),
    ]
    return build_mlir(fname, [("input", A)], lines, C)

def _gen_reshape_same_rank(fname, A, C):
    nd = len(A); tA = tensor_type(A); tC = tensor_type(C)
    used = [False] * nd; perm = []
    for cv in C:
        best = None
        for ai, av in enumerate(A):
            if not used[ai] and av == cv: best = ai; break
        if best is None: return _reshape_placeholder(fname, A, C)
        perm.append(best); used[best] = True
    perm_inv = [0] * nd
    for oi, ai in enumerate(perm): perm_inv[ai] = oi
    body = _consts(max(2, nd))
    a_lines, a_d = _dims("input", A); body += a_lines
    out_dv = {i: a_d[perm[i]] for i in range(nd)}
    body.append(_empty(C, out_dv))
    lvars = ["%rsi{}".format(i) for i in range(nd)]; out_idx = ", ".join(lvars)
    in_idx = ", ".join(lvars[perm_inv[i]] for i in range(nd))
    carry = "%out"; t = tC
    for depth in range(nd):
        v = lvars[depth]; hi = a_d[perm[depth]]
        res = "%t_res_{}".format(v.lstrip("%"))
        body.append("    {} = scf.for {} = %c0 to {} step %c1 iter_args(%t_rs_{} = {}) -> ({}) {{".format(
            res, v, hi, v.lstrip("%"), carry, t))
        carry = "%t_rs_{}".format(v.lstrip("%"))
    body.append("    %rsv = tensor.extract %input[{}] : {}".format(in_idx, tA))
    body.append("    %t_ins = tensor.insert %rsv into {}[{}] : {}".format(carry, out_idx, tC))
    body.append("    scf.yield %t_ins : {}".format(tC))
    for depth in range(nd - 1, -1, -1):
        v = lvars[depth]; res = "%t_res_{}".format(v.lstrip("%"))
        body.append("    }")
        if depth > 0:
            body.append("    scf.yield {} : {}".format(res, tC))
    body.append("    return {} : {}".format("%t_res_{}".format(lvars[0].lstrip("%")), tC))
    return build_mlir(fname, [("input", A)], body, C)

def gen_reshape(stem, shapes):
    if len(shapes) < 2: return None
    A, C = shapes[0], shapes[1]; fname = stem
    ndA, ndC = len(A), len(C)
    if ndC > ndA: return _gen_reshape_expand(fname, A, C)
    elif ndC < ndA: return _gen_reshape_collapse(fname, A, C)
    else: return _gen_reshape_same_rank(fname, A, C)

# ---- concat ----

def gen_concat(stem, shapes):
    if len(shapes) < 3: return None
    inputs = shapes[:-1]; C = shapes[-1]
    fname = stem; nd = len(C); n_inp = len(inputs)
    axis = _find_concat_axis(inputs, C)
    body = _consts(max(2, nd))
    all_dvars = []
    for ii, inp in enumerate(inputs):
        lines, dv = _dims("in{}".format(ii), inp); body += lines; all_dvars.append(dv)
    eq_cnt = 0
    for ii in range(1, n_inp):
        for d in range(nd):
            if d == axis: continue
            lines, eq_cnt = _assert_eq(all_dvars[0][d], all_dvars[ii][d],
                "in0.dim({})==in{}.dim({})".format(d, ii, d), eq_cnt)
            body += lines
    out_dv = {}
    for d in range(nd):
        if d != axis: out_dv[d] = all_dvars[0][d]
    if C[axis] == "?":
        sum_var = "%c0"
        for ii in range(n_inp):
            tmp = "%csum{}".format(ii)
            body.append("    {} = arith.addi {}, {} : index".format(tmp, sum_var, all_dvars[ii][axis]))
            sum_var = tmp
        out_dv[axis] = sum_var
    body.append(_empty(C, out_dv))
    tC = tensor_type(C)
    # Copy each input sequentially, each pass updates the tensor
    carry_name = "%out"
    cum_off_vars = {}
    for ii, inp in enumerate(inputs):
        t_inp = tensor_type(inp); dv_ii = all_dvars[ii]
        lvars = ["%cc{}_{}".format(ii, d) for d in range(nd)]
        lvars_str = ", ".join(lvars)
        # Precompute cumulative axis offsets outside all loops
        if ii == 0:
            cum_off = "%c0"  # offset for input 0 is always 0
        else:
            cum_var = "%cum_off_{}".format(ii)
            # Build cumulative sum of all previous input axis dims
            prev_cum = "%cum_off_{}".format(ii - 1) if ii > 1 else "%in0_d{}".format(axis)
            cur_sz   = "%in{}_d{}".format(ii - 1, axis)
            if ii == 1:
                # cum_off_1 = in0_d{axis}
                cum_off = "%in0_d{}".format(axis)
            else:
                # cum_off_ii = prev_cum + in{ii-1}_d{axis}
                # prev_cum was already computed in the previous iter and stored in cum_off_vars
                body.append("    {} = arith.addi {}, {} : index".format(
                    cum_var, cum_off_vars[ii - 1], "%in{}_d{}".format(ii - 1, axis)))
                cum_off = cum_var
        cum_off_vars[ii] = cum_off
        # Build nested loops for this input
        carry = carry_name; t = tC
        for depth in range(nd):
            v = lvars[depth]; hi = dv_ii[depth]
            res = "%t_cc{}_d{}".format(ii, depth)
            body.append("    {} = scf.for {} = %c0 to {} step %c1 iter_args(%t_cc{}_{} = {}) -> ({}) {{".format(
                res, v, hi, ii, depth, carry, t))
            carry = "%t_cc{}_{}".format(ii, depth)
        # Compute output axis index with offset
        if ii == 0:
            out_axis_var = lvars[axis]
        else:
            off_var = "%t_ax_off{}".format(ii)
            body.append("    {} = arith.addi {}, {} : index".format(off_var, cum_off, lvars[axis]))
            out_axis_var = off_var
        out_parts = list(lvars); out_parts[axis] = out_axis_var
        out_idx_str = ", ".join(out_parts)
        body.append("    %cv{} = tensor.extract %in{}[{}] : {}".format(ii, ii, lvars_str, t_inp))
        body.append("    %t_ins_cc{} = tensor.insert %cv{} into {}[{}] : {}".format(ii, ii, carry, out_idx_str, tC))
        body.append("    scf.yield %t_ins_cc{} : {}".format(ii, tC))
        for depth in range(nd - 1, -1, -1):
            res = "%t_cc{}_d{}".format(ii, depth)
            body.append("    }")
            if depth > 0:
                body.append("    scf.yield {} : {}".format(res, tC))
        carry_name = "%t_cc{}_d0".format(ii)
    body.append("    return {} : {}".format(carry_name, tC))
    arg_list = [("in{}".format(ii), inp) for ii, inp in enumerate(inputs)]
    return build_mlir(fname, arg_list, body, C)

# ---- batch_norm ----

def gen_batch_norm(stem, shapes):
    if len(shapes) < 3: return None
    inp = shapes[0]; C_out = shapes[-1]; nd = len(inp)
    gamma_dims = shapes[1]; beta_dims = shapes[2] if len(shapes) >= 4 else shapes[1]
    ch_axis = 1
    for ai, av in enumerate(inp):
        if av == gamma_dims[0]: ch_axis = ai; break
    fname = stem; body = _consts(max(2, nd))
    a_lines, a_d = _dims("input", inp)
    g_lines, g_d = _dims("gamma", gamma_dims)
    b_lines, b_d = _dims("beta",  beta_dims)
    body += a_lines + g_lines + b_lines
    eq_cnt = 0
    lines, eq_cnt = _assert_eq(g_d[0], a_d[ch_axis], "gamma.dim(0)==input.dim({})".format(ch_axis), eq_cnt); body += lines
    lines, eq_cnt = _assert_eq(b_d[0], a_d[ch_axis], "beta.dim(0)==input.dim({})".format(ch_axis), eq_cnt); body += lines
    body.append(_empty(C_out, {i: a_d[i] for i in range(nd) if inp[i] == "?"}))
    tA = tensor_type(inp); tG = tensor_type(gamma_dims); tBt = tensor_type(beta_dims); tC = tensor_type(C_out)
    lvars = ["%bn{}".format(i) for i in range(nd)]; idxs = ", ".join(lvars)
    carry = "%out"; t = tC
    for depth in range(nd):
        v = lvars[depth]; hi = a_d[depth]
        res = "%t_res_{}".format(v.lstrip("%"))
        body.append("    {} = scf.for {} = %c0 to {} step %c1 iter_args(%t_bn_{} = {}) -> ({}) {{".format(
            res, v, hi, v.lstrip("%"), carry, t))
        carry = "%t_bn_{}".format(v.lstrip("%"))
    body += [
        "    %xv  = tensor.extract %input[{}] : {}".format(idxs, tA),
        "    %gv  = tensor.extract %gamma[{}] : {}".format(lvars[ch_axis], tG),
        "    %bv  = tensor.extract %beta[{}]  : {}".format(lvars[ch_axis], tBt),
        "    %scl = arith.mulf %xv, %gv : f32",
        "    %res = arith.addf %scl, %bv : f32",
        "    %t_ins = tensor.insert %res into {}[{}] : {}".format(carry, idxs, tC),
        "    scf.yield %t_ins : {}".format(tC),
    ]
    for depth in range(nd - 1, -1, -1):
        v = lvars[depth]; res = "%t_res_{}".format(v.lstrip("%"))
        body.append("    }")
        if depth > 0:
            body.append("    scf.yield {} : {}".format(res, tC))
    body.append("    return {} : {}".format("%t_res_{}".format(lvars[0].lstrip("%")), tC))
    return build_mlir(fname, [("input", inp), ("gamma", gamma_dims), ("beta", beta_dims)], body, C_out)

# ---- layer_normalization ----

def gen_layer_norm(stem, shapes):
    if len(shapes) < 2: return None
    inp = shapes[0]; nd = len(inp); C_out = inp
    gamma_dims = shapes[1]; beta_dims = shapes[2] if len(shapes) >= 3 else gamma_dims
    norm_rank  = len(gamma_dims); batch_rank = max(0, nd - norm_rank)
    fname = stem; body = _consts(max(2, nd))
    a_lines, a_d = _dims("input", inp)
    g_lines, g_d = _dims("gamma", gamma_dims)
    b_lines, b_d = _dims("beta",  beta_dims)
    body += a_lines + g_lines + b_lines
    eq_cnt = 0
    for ni in range(norm_rank):
        lines, eq_cnt = _assert_eq(g_d[ni], a_d[batch_rank + ni],
            "gamma.dim({})==input.dim({})".format(ni, batch_rank + ni), eq_cnt); body += lines
    for ni in range(norm_rank):
        lines, eq_cnt = _assert_eq(b_d[ni], a_d[batch_rank + ni],
            "beta.dim({})==input.dim({})".format(ni, batch_rank + ni), eq_cnt); body += lines
    body.append(_empty(C_out, {i: a_d[i] for i in range(nd) if inp[i] == "?"}))
    tA = tensor_type(inp); tG = tensor_type(gamma_dims); tBt = tensor_type(beta_dims); tC = tensor_type(C_out)
    batch_lvars = ["%ln_b{}".format(i) for i in range(batch_rank)]
    norm_lvars  = ["%ln_n{}".format(i) for i in range(norm_rank)]
    all_idxs    = ", ".join(batch_lvars + norm_lvars); norm_idxs = ", ".join(norm_lvars)
    norm_total  = 1; norm_total_dyn = False
    for ni in range(norm_rank):
        d = inp[batch_rank + ni]
        if isinstance(d, int): norm_total *= d
        else: norm_total_dyn = True; break
    norm_specs = [(norm_lvars[ni], a_d[batch_rank + ni]) for ni in range(norm_rank)]

    if norm_total_dyn:
        body.append("    %nsz_idx_0 = arith.constant 1 : index")
        for ni in range(norm_rank):
            body.append("    %nsz_idx_{} = arith.muli %nsz_idx_{}, {} : index".format(
                ni + 1, ni, a_d[batch_rank + ni]))
        body.append("    %nsz_i64 = arith.index_cast %nsz_idx_{} : index to i64".format(norm_rank))
        body.append("    %nsz = arith.sitofp %nsz_i64 : i64 to f32")
    else:
        body.append("    %nsz = arith.constant {} : f32".format(float(norm_total)))

    # Scalar accumulators via 0-d tensors
    body += [
        "    %acc_sum_t   = tensor.empty() : tensor<f32>",
        "    %acc_sumsq_t = tensor.empty() : tensor<f32>",
    ]

    # Outer batch loops carry the output tensor
    carry = "%out"; t = tC
    for depth in range(batch_rank):
        v = batch_lvars[depth]; hi = a_d[depth]
        res = "%t_res_{}".format(v.lstrip("%"))
        body.append("    {} = scf.for {} = %c0 to {} step %c1 iter_args(%t_ln_{} = {}) -> ({}) {{".format(
            res, v, hi, v.lstrip("%"), carry, t))
        carry = "%t_ln_{}".format(v.lstrip("%"))

    # Passes 1 & 2 use scalar iter_args over norm dims
    body += [
        "    %zero_sf = arith.constant 0.0 : f32",
        "    %eps     = arith.constant 1.0e-05 : f32",
    ]
    # Pass 1: sum - fully nested iter_args for all norm dims
    _ln_sum_carry = "%zero_sf"
    for depth in range(norm_rank):
        v, hi = norm_specs[depth]
        _ln_arg = "%ln_sa{}".format(depth)
        _ln_res = "%ln_sum" if depth == 0 else "%ln_sum_d{}".format(depth)
        body.append("    {} = scf.for {} = %c0 to {} step %c1 iter_args({} = {}) -> (f32) {{".format(
            _ln_res, v, hi, _ln_arg, _ln_sum_carry))
        _ln_sum_carry = _ln_arg
    body += [
        "    %lnv1 = tensor.extract %input[{}] : {}".format(all_idxs, tA),
        "    %ns1  = arith.addf {}, %lnv1 : f32".format(_ln_sum_carry),
        "    scf.yield %ns1 : f32",
    ]
    for depth in range(norm_rank - 1, -1, -1):
        _ln_res = "%ln_sum" if depth == 0 else "%ln_sum_d{}".format(depth)
        body.append("    }")
        if depth > 0:
            body.append("    scf.yield {} : f32".format(_ln_res))
    # Pass 2: sum of squares - fully nested iter_args for all norm dims
    _ln_ssq_carry = "%zero_sf"
    for depth in range(norm_rank):
        v, hi = norm_specs[depth]
        _ln_ssa = "%ln_ssa{}".format(depth)
        _ln_ssq_res = "%ln_sumsq" if depth == 0 else "%ln_sumsq_d{}".format(depth)
        body.append("    {} = scf.for {} = %c0 to {} step %c1 iter_args({} = {}) -> (f32) {{".format(
            _ln_ssq_res, v, hi, _ln_ssa, _ln_ssq_carry))
        _ln_ssq_carry = _ln_ssa
    body += [
        "    %lnv2 = tensor.extract %input[{}] : {}".format(all_idxs, tA),
        "    %sq   = arith.mulf %lnv2, %lnv2 : f32",
        "    %ssq  = arith.addf {}, %sq : f32".format(_ln_ssq_carry),
        "    scf.yield %ssq : f32",
    ]
    for depth in range(norm_rank - 1, -1, -1):
        _ln_ssq_res = "%ln_sumsq" if depth == 0 else "%ln_sumsq_d{}".format(depth)
        body.append("    }")
        if depth > 0:
            body.append("    scf.yield {} : f32".format(_ln_ssq_res))
    # Stats
    body += [
        "    %mean    = arith.divf %ln_sum, %nsz : f32",
        "    %msq     = arith.mulf %mean, %mean : f32",
        "    %esq     = arith.divf %ln_sumsq, %nsz : f32",
        "    %var     = arith.subf %esq, %msq : f32",
        "    %vep     = arith.addf %var, %eps : f32",
        "    %std     = math.sqrt %vep : f32",
        "    %one_f   = arith.constant 1.0 : f32",
        "    %inv_std = arith.divf %one_f, %std : f32",
    ]
    # Pass 3: normalize+scale via tensor iter_args over norm dims
    norm_carry = carry; nt = tC
    for depth in range(norm_rank):
        v, hi = norm_specs[depth]
        res = "%t_res_ln_n{}".format(depth)
        body.append("    {} = scf.for {} = %c0 to {} step %c1 iter_args(%t_ln_n{} = {}) -> ({}) {{".format(
            res, v, hi, depth, norm_carry, nt))
        norm_carry = "%t_ln_n{}".format(depth)
    body += [
        "    %lnv3   = tensor.extract %input[{}] : {}".format(all_idxs, tA),
        "    %gv     = tensor.extract %gamma[{}] : {}".format(norm_idxs, tG),
        "    %bv     = tensor.extract %beta[{}]  : {}".format(norm_idxs, tBt),
        "    %cent   = arith.subf %lnv3, %mean : f32",
        "    %normed = arith.mulf %cent, %inv_std : f32",
        "    %scaled = arith.mulf %normed, %gv : f32",
        "    %res    = arith.addf %scaled, %bv : f32",
        "    %t_ins_ln = tensor.insert %res into {}[{}] : {}".format(norm_carry, all_idxs, tC),
        "    scf.yield %t_ins_ln : {}".format(tC),
    ]
    last_norm_res = "%t_res_ln_n0"  # outermost norm loop result
    for depth in range(norm_rank - 1, -1, -1):
        res = "%t_res_ln_n{}".format(depth)
        body.append("    }")
        if depth > 0:
            body.append("    scf.yield {} : {}".format(res, tC))
    # Close batch loops
    last_res = last_norm_res
    for depth in range(batch_rank - 1, -1, -1):
        v = batch_lvars[depth]; res = "%t_res_{}".format(v.lstrip("%"))
        body.append("    scf.yield {} : {}".format(last_res, tC))
        last_res = res
        body.append("    }")
    body.append("    return {} : {}".format(last_res, tC))
    return build_mlir(fname, [("input", inp), ("gamma", gamma_dims), ("beta", beta_dims)], body, C_out)

# ---- conv2d ----

def gen_conv2d(stem, shapes):
    if len(shapes) < 3: return None
    inp, filt, out = shapes[0], shapes[1], shapes[2]; fname = stem
    strides = 1
    if len(inp) == 4 and len(out) == 4 and isinstance(inp[2], int) and isinstance(out[2], int) and out[2] > 0:
        strides = max(1, inp[2] // out[2])
    kH = filt[2] if len(filt) == 4 and isinstance(filt[2], int) else 3
    kW = filt[3] if len(filt) == 4 and isinstance(filt[3], int) else 3
    body = _consts(max(2, len(inp)))
    body.append("    %zero_f = arith.constant 0.0 : f32")
    a_lines, a_d = _dims("input", inp); f_lines, f_d = _dims("filter", filt)
    body += a_lines + f_lines
    eq_cnt = 0
    lines, eq_cnt = _assert_eq(a_d[1], f_d[1], "input.dim(1)==filter.dim(1)", eq_cnt); body += lines
    tA = tensor_type(inp); tF = tensor_type(filt); tC = tensor_type(out)

    def _out_sz(i):
        d = out[i]
        if isinstance(d, int):
            v = "%out_sz{}".format(i); body.append("    {} = arith.constant {} : index".format(v, d)); return v
        if i == 0: return a_d[0]
        if i == 1: return f_d[0]
        return a_d[i]

    out_dv = {i: _out_sz(i) for i in range(len(out)) if out[i] == "?"}
    body.append(_empty(out, out_dv))
    body += [
        "    %stride = arith.constant {} : index".format(strides),
        "    %kH_sz  = arith.constant {} : index".format(kH),
        "    %kW_sz  = arith.constant {} : index".format(kW),
    ]
    oh_hi = _out_sz(2); ow_hi = _out_sz(3)
    n_hi = a_d[0]; f_hi = f_d[0]; c_hi = a_d[1]
    # Outer 4 loops carry tensor; inner 3 loops do the k-reduction as scalar then insert
    carry = "%out"; t = tC
    for lv, hi in [("%n", n_hi), ("%f", f_hi), ("%oh", oh_hi), ("%ow", ow_hi)]:
        res = "%t_res_{}".format(lv.lstrip("%"))
        body.append("    {} = scf.for {} = %c0 to {} step %c1 iter_args(%t_cv_{} = {}) -> ({}) {{".format(
            res, lv, hi, lv.lstrip("%"), carry, t))
        carry = "%t_cv_{}".format(lv.lstrip("%"))
    # k-reduction as scalar iter_args
    body.append("    %acc = scf.for %c = %c0 to {} step %c1 iter_args(%s_c = %zero_f) -> (f32) {{".format(c_hi))
    body.append("    %acc2 = scf.for %kh = %c0 to %kH_sz step %c1 iter_args(%s_kh = %s_c) -> (f32) {")
    body.append("    %acc3 = scf.for %kw = %c0 to %kW_sz step %c1 iter_args(%s_kw = %s_kh) -> (f32) {")
    body += [
        "    %s_oh = arith.muli %stride, %oh : index",
        "    %ih   = arith.addi %s_oh, %kh : index",
        "    %s_ow = arith.muli %stride, %ow : index",
        "    %iw   = arith.addi %s_ow, %kw : index",
        "    %xv   = tensor.extract %input[%n, %c, %ih, %iw]  : {}".format(tA),
        "    %wv   = tensor.extract %filter[%f, %c, %kh, %kw] : {}".format(tF),
        "    %ml   = arith.mulf %xv, %wv : f32",
        "    %sm   = arith.addf %s_kw, %ml : f32",
        "    scf.yield %sm : f32",
        "    }",
        "    scf.yield %acc3 : f32",
        "    }",
        "    scf.yield %acc2 : f32",
        "    }",
        "    %t_new = tensor.insert %acc into {}[%n, %f, %oh, %ow] : {}".format(carry, tC),
        "    scf.yield %t_new : {}".format(tC),
    ]
    for lv in ["%ow", "%oh", "%f", "%n"]:
        res = "%t_res_{}".format(lv.lstrip("%"))
        body.append("    }")
        inner = {"ow": "oh", "oh": "f", "f": "n"}
        if lv.lstrip("%") in inner:
            body.append("    scf.yield {} : {}".format("%t_res_{}".format(lv.lstrip("%")), tC))
    body.append("    return %t_res_n : {}".format(tC))
    return build_mlir(fname, [("input", inp), ("filter", filt)], body, out)

# ---- max_pool2d ----

def gen_max_pool2d(stem, shapes):
    if len(shapes) < 2: return None
    inp, out = shapes[0], shapes[1]
    if len(inp) < 4 or len(out) < 4: return None
    fname = stem
    strides = 2
    if isinstance(inp[2], int) and isinstance(out[2], int) and out[2] > 0:
        strides = max(1, inp[2] // out[2])
    kH_val = kW_val = strides
    m = re.search(r"Hd(\d+)", stem)
    if m: strides = int(m.group(1)); kH_val = kW_val = strides
    body = _consts(max(2, len(inp)))
    body.append("    %neg_inf = arith.constant -3.4028234663852886e+38 : f32")
    a_lines, a_d = _dims("input", inp); body += a_lines

    def _out_sz(i):
        d = out[i]
        if isinstance(d, int):
            v = "%po_sz{}".format(i); body.append("    {} = arith.constant {} : index".format(v, d)); return v
        return a_d[i]

    out_dv = {i: _out_sz(i) for i in range(len(out)) if out[i] == "?"}
    body.append(_empty(out, out_dv))
    body += [
        "    %kh_sz  = arith.constant {} : index".format(kH_val),
        "    %kw_sz  = arith.constant {} : index".format(kW_val),
        "    %stride = arith.constant {} : index".format(strides),
    ]
    tA = tensor_type(inp); tC = tensor_type(out)
    oh_hi = _out_sz(2); ow_hi = _out_sz(3)
    carry = "%out"; t = tC
    for lv, hi in [("%pn", a_d[0]), ("%pc", a_d[1]), ("%poh", oh_hi), ("%pow", ow_hi)]:
        res = "%t_res_{}".format(lv.lstrip("%"))
        body.append("    {} = scf.for {} = %c0 to {} step %c1 iter_args(%t_mp_{} = {}) -> ({}) {{".format(
            res, lv, hi, lv.lstrip("%"), carry, t))
        carry = "%t_mp_{}".format(lv.lstrip("%"))
    # Kernel loops as scalar iter_args (max reduction)
    body.append("    %pool = scf.for %pkh = %c0 to %kh_sz step %c1 iter_args(%mp_kh = %neg_inf) -> (f32) {")
    body.append("    %pool2 = scf.for %pkw = %c0 to %kw_sz step %c1 iter_args(%mp_kw = %mp_kh) -> (f32) {")
    body += [
        "    %s_poh = arith.muli %stride, %poh : index",
        "    %ih    = arith.addi %s_poh, %pkh : index",
        "    %s_pow = arith.muli %stride, %pow : index",
        "    %iw    = arith.addi %s_pow, %pkw : index",
        "    %pv    = tensor.extract %input[%pn, %pc, %ih, %iw] : {}".format(tA),
        "    %gt    = arith.cmpf ogt, %pv, %mp_kw : f32",
        "    %mx    = arith.select %gt, %pv, %mp_kw : f32",
        "    scf.yield %mx : f32",
        "    }",
        "    scf.yield %pool2 : f32",
        "    }",
        "    %t_new = tensor.insert %pool into {}[%pn, %pc, %poh, %pow] : {}".format(carry, tC),
        "    scf.yield %t_new : {}".format(tC),
    ]
    for lv in ["%pow", "%poh", "%pc", "%pn"]:
        res = "%t_res_{}".format(lv.lstrip("%"))
        body.append("    }")
        inner = {"pow": "poh", "poh": "pc", "pc": "pn"}
        if lv.lstrip("%") in inner:
            body.append("    scf.yield {} : {}".format(res, tC))
    body.append("    return %t_res_pn : {}".format(tC))
    return build_mlir(fname, [("input", inp)], body, out)

# ---- reduce_mean ----

def gen_reduce_mean(stem, shapes):
    if len(shapes) < 2: return None
    inp, out_declared = shapes[0], shapes[-1]; ndA = len(inp)
    reduce_dims, extra_out = find_reduce_dims(inp, out_declared)
    if not reduce_dims: reduce_dims = [ndA - 1]; extra_out = []
    kept    = [i for i in range(ndA) if i not in set(reduce_dims)]
    out_mid = [inp[i] for i in kept]; ndMid = len(out_mid)
    fname = stem; body = _consts(max(2, ndA))
    body.append("    %zero_f = arith.constant 0.0 : f32")
    a_lines, a_d = _dims("input", inp); body += a_lines
    tInp = tensor_type(inp); tMid = tensor_type(out_mid) if out_mid else "tensor<f32>"
    mid_dv = {i: a_d[kept[i]] for i in range(ndMid) if out_mid[i] == "?"}
    body.append(_empty(out_mid, mid_dv, name="%sum_buf") if out_mid else "    %sum_buf = tensor.empty() : tensor<f32>")
    kept_lvars = ["%rm_k{}".format(i) for i in range(ndMid)]
    red_lvars  = ["%rm_r{}".format(i) for i in range(len(reduce_dims))]
    kept_specs = [(kept_lvars[i], a_d[kept[i]]) for i in range(ndMid)]
    red_specs  = [(red_lvars[i], a_d[reduce_dims[i]]) for i in range(len(reduce_dims))]
    all_in = [""] * ndA
    for ki, ai in enumerate(kept): all_in[ai] = kept_lvars[ki]
    for ri, ai in enumerate(reduce_dims): all_in[ai] = red_lvars[ri]
    all_in_idxs = ", ".join(all_in)
    kept_idxs = ", ".join(kept_lvars) if kept_lvars else ""

    # Outer kept-dim loops carry the sum_buf tensor
    carry_sb = "%sum_buf"; tsb = tMid
    for depth in range(ndMid):
        v, hi = kept_specs[depth]
        res = "%t_rm_k{}".format(depth)
        body.append("    {} = scf.for {} = %c0 to {} step %c1 iter_args(%t_sb_{} = {}) -> ({}) {{".format(
            res, v, hi, v.lstrip("%"), carry_sb, tsb))
        carry_sb = "%t_sb_{}".format(v.lstrip("%"))
    # Inner reduce loops - fully nested iter_args for all reduce dims
    _red_carry = "%zero_f"
    for ri, (rv, rh) in enumerate(red_specs):
        _red_arg = "%ia{}".format(ri)
        _red_res = "%red_val" if ri == 0 else "%red_val_d{}".format(ri)
        body.append("    {} = scf.for {} = %c0 to {} step %c1 iter_args({} = {}) -> (f32) {{".format(
            _red_res, rv, rh, _red_arg, _red_carry))
        _red_carry = _red_arg
    body += [
        "    %rv = tensor.extract %input[{}] : {}".format(all_in_idxs, tInp),
        "    %ns = arith.addf {}, %rv : f32".format(_red_carry),
        "    scf.yield %ns : f32",
    ]
    for ri in range(len(reduce_dims) - 1, -1, -1):
        _red_res = "%red_val" if ri == 0 else "%red_val_d{}".format(ri)
        body.append("    }")
        if ri > 0:
            body.append("    scf.yield {} : f32".format(_red_res))
    # Insert reduced value into sum_buf
    if kept_idxs:
        body.append("    %t_sb_new = tensor.insert %red_val into {}[{}] : {}".format(carry_sb, kept_idxs, tMid))
        body.append("    scf.yield %t_sb_new : {}".format(tMid))
    else:
        body.append("    %t_sb_new = tensor.insert %red_val into {} : tensor<f32>".format(carry_sb))
        body.append("    scf.yield %t_sb_new : tensor<f32>")
    for depth in range(ndMid - 1, -1, -1):
        v = kept_lvars[depth]; res = "%t_rm_k{}".format(depth)
        body.append("    }")
        if depth > 0: body.append("    scf.yield {} : {}".format(res, tMid))

    red_size = 1
    for d in reduce_dims:
        if isinstance(inp[d], int): red_size *= inp[d]
    scale_val = (1.0 / red_size) if red_size > 1 else 1.0
    sum_result = "%t_rm_k0" if ndMid > 0 else "%t_sb_new"
    body += ["    %scale = arith.constant {} : f32".format(scale_val)]
    body.append(_empty(out_mid, mid_dv, name="%out") if out_mid else "    %out = tensor.empty() : tensor<f32>")
    carry_out = "%out"; tout = tMid
    for depth in range(ndMid):
        v, hi = kept_specs[depth]
        res = "%t_rm_sc{}".format(depth)
        body.append("    {} = scf.for {} = %c0 to {} step %c1 iter_args(%t_sc_{} = {}) -> ({}) {{".format(
            res, v.replace("rm_k", "sc"), hi, v.lstrip("%"), carry_out, tout))
        carry_out = "%t_sc_{}".format(v.lstrip("%"))
    sc_lvars = [v.replace("rm_k", "sc") for v in kept_lvars]
    sc_idxs  = ", ".join(sc_lvars) if sc_lvars else ""
    if sc_idxs:
        body += [
            "    %sv = tensor.extract {}[{}] : {}".format(sum_result, sc_idxs, tMid),
            "    %me = arith.mulf %sv, %scale : f32",
            "    %t_sc_ins = tensor.insert %me into {}[{}] : {}".format(carry_out, sc_idxs, tMid),
            "    scf.yield %t_sc_ins : {}".format(tMid),
        ]
    else:
        body += [
            "    %sv = tensor.extract {} [] : tensor<f32>".format(sum_result),
            "    %me = arith.mulf %sv, %scale : f32",
            "    %t_sc_ins = tensor.insert %me into {} : tensor<f32>".format(carry_out),
            "    scf.yield %t_sc_ins : tensor<f32>",
        ]
    for depth in range(ndMid - 1, -1, -1):
        res = "%t_rm_sc{}".format(depth)
        body.append("    }")
        if depth > 0: body.append("    scf.yield {} : {}".format(res, tMid))
    out_mid_res = "%t_rm_sc0" if ndMid > 0 else "%t_sc_ins"
    if extra_out:
        nextra = len(extra_out)
        if ndMid > 0:
            reassoc = [[i] for i in range(ndMid - 1)] + [list(range(ndMid - 1, ndMid + nextra))]
        else:
            reassoc = [list(range(nextra))]
        reassoc_str = "[" + ", ".join("[" + ", ".join(str(x) for x in g) + "]" for g in reassoc) + "]"
        tFinal = tensor_type(out_declared)
        out_shape_str = "[" + ", ".join(str(d) for d in out_declared) + "]"
        body += [
            "    %result = tensor.expand_shape {} {} output_shape {} : {} into {}".format(
                out_mid_res, reassoc_str, out_shape_str, tMid, tFinal),
            "    return %result : {}".format(tFinal),
        ]
        return build_mlir(fname, [("input", inp)], body, out_declared)
    else:
        body.append("    return {} : {}".format(out_mid_res, tMid))
        return build_mlir(fname, [("input", inp)], body, out_mid)

# ---- embedding ----

def gen_embedding(stem, shapes):
    if len(shapes) < 3: return None
    idx_dims, tbl_dims, out_dims = shapes[0], shapes[1], shapes[2]
    nd_idx = len(idx_dims); nd_tbl = len(tbl_dims); nd_out = len(out_dims); fname = stem
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
    body.append(_empty(out_dims, out_dv))
    tIdx = tensor_type(idx_dims, "i64"); tTbl = tensor_type(tbl_dims); tOut = tensor_type(out_dims)
    b_lvars = ["%em_b{}".format(i) for i in range(nd_idx)]
    d_lvars = ["%em_d{}".format(i) for i in range(nd_tbl - 1)]
    b_idxs  = ", ".join(b_lvars); d_idxs = ", ".join(d_lvars)
    all_out = ", ".join(b_lvars + d_lvars); tbl_idxs = "%row_idx, " + d_idxs
    carry = "%out"; t = tOut
    all_specs = [(b_lvars[i], a_d[i]) for i in range(nd_idx)] + \
                [(d_lvars[i], t_d[i + 1]) for i in range(nd_tbl - 1)]
    for depth, (v, hi) in enumerate(all_specs):
        res = "%t_em_d{}".format(depth)
        body.append("    {} = scf.for {} = %c0 to {} step %c1 iter_args(%t_em_{} = {}) -> ({}) {{".format(
            res, v, hi, v.lstrip("%"), carry, t))
        carry = "%t_em_{}".format(v.lstrip("%"))
    body += [
        "    %raw_idx = tensor.extract %indices[{}] : {}".format(b_idxs, tIdx),
        "    %row_idx = arith.index_cast %raw_idx : i64 to index",
        "    %tv      = tensor.extract %table[{}] : {}".format(tbl_idxs, tTbl),
        "    %t_ins   = tensor.insert %tv into {}[{}] : {}".format(carry, all_out, tOut),
        "    scf.yield %t_ins : {}".format(tOut),
    ]
    for depth in range(len(all_specs) - 1, -1, -1):
        res = "%t_em_d{}".format(depth)
        body.append("    }")
        if depth > 0: body.append("    scf.yield {} : {}".format(res, tOut))
    body.append("    return %t_em_d0 : {}".format(tOut))
    return build_mlir(fname, [("indices", idx_dims, "i64"), ("table", tbl_dims)], body, out_dims)

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

GENERATORS = {
    "matmul":             gen_matmul,
    "elemwise_add":       gen_elemwise_add,
    "relu":               gen_relu,
    "sigmoid":            gen_sigmoid,
    "gelu":               gen_gelu,
    "softmax":            gen_softmax,
    "transpose":          gen_transpose,
    "reshape":            gen_reshape,
    "concat":             gen_concat,
    "batch_norm":         gen_batch_norm,
    "layer_normalization": gen_layer_norm,
    "conv2d":             gen_conv2d,
    "max_pool2d":         gen_max_pool2d,
    "reduce_mean":        gen_reduce_mean,
    "embedding":          gen_embedding,
}

def generate_for_file(co_path, category, dry_run=False, verbose=False):
    stem   = co_path.stem
    shapes = shapes_for_category(stem, category)
    gen_fn = GENERATORS.get(category)
    if gen_fn is None:
        print("  [SKIP] No generator for category {}".format(category)); return False
    mlir_code = gen_fn(stem, shapes)
    if mlir_code is None:
        print("  [SKIP] Generator returned None for {}".format(stem)); return False
    out_path = MLIR_DIR / category / (stem + ".mlir")
    if verbose: print("  -> {}".format(out_path.relative_to(ROOT)))
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(mlir_code)
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--category", default=None)
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()
    categories = sorted(c.name for c in CHOREO_DIR.iterdir() if c.is_dir())
    if args.category: categories = [args.category]
    total, ok = 0, 0
    for cat in categories:
        co_files = sorted((CHOREO_DIR / cat).glob("*.co"))
        if not co_files: continue
        if args.verbose: print("\n=== {} ({} files) ===".format(cat, len(co_files)))
        for cf in co_files:
            total += 1
            if generate_for_file(cf, cat, dry_run=args.dry_run, verbose=args.verbose): ok += 1
    print("\nGenerated {}/{} files{}".format(ok, total, " (dry-run)" if args.dry_run else ""))

if __name__ == "__main__":
    main()
