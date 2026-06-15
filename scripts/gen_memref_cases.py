#!/usr/bin/env python3
"""gen_memref_cases.py — generate memref-level MLIR benchmarks with cf.assert.

Each generated file implements one benchmark case using the memref dialect:
  • memref.alloc()       — allocate output buffer
  • memref.load/store    — element access
  • memref.dim           — runtime dimension queries
  • cf.assert            — explicit shape invariants (the set of constraints
                           a manually-written implementation would add at the
                           memref level; used to compare against Choreo's
                           automatically generated assertions)
  • scf.for              — loop iteration (no iter_args on outer loops;
                           scalar iter_args for inner reductions)

Dynamic-dim cases (those in benchmark/choreo/ whose stem contains '?') are
skipped — matching the same static-only filter as gen_stablehlo_cases.py.

Output: benchmark/memref/{category}/{stem}.mlir
"""

import math as _math
from itertools import permutations
from pathlib import Path

ROOT         = Path(__file__).parent.parent
STABLEHLO_DIR = ROOT / "benchmark/stablehlo"   # iterate exactly this set
OUT_DIR      = ROOT / "benchmark/memref"

# ---------------------------------------------------------------------------
# Shape helpers (mirrors gen_stablehlo_cases.py)
# ---------------------------------------------------------------------------

def _to_dim(s):
    try:
        return int(s)
    except ValueError:
        return "?"

def parse_shape(s):
    return [_to_dim(p) for p in s.split("x") if p]

def is_static(dims):
    return all(isinstance(d, int) for d in dims)

def mt(dims, dtype="f32"):
    """Return memref<d0xd1x...xdtype> string."""
    return "memref<{}x{}>".format("x".join(str(d) for d in dims), dtype)

def _parts_after_index_name(stem):
    parts = stem.split("_")
    idx = 1
    while idx < len(parts):
        p = parts[idx]
        if p.isalpha() and p.islower():
            idx += 1; continue
        if "x" in p:
            break
        if p and (p[0].isdigit() or p[0].isupper()):
            break
        idx += 1
    return parts[idx:]

def shapes_for_category(stem, category):
    raw = _parts_after_index_name(stem)
    if category == "conv2d":
        shapes = [parse_shape(r) for r in raw if "x" in r]
        return shapes[:3]
    if category == "max_pool2d":
        shapes = [parse_shape(r) for r in raw]
        return [s for s in shapes if len(s) >= 2]
    return [parse_shape(r) for r in raw]

def _find_bcast_dims(src_dims, out_dims):
    """Map each src axis to a matching out axis by value (first unused match)."""
    used = [False] * len(out_dims)
    result = []
    for sd in src_dims:
        matched = False
        for i, od in enumerate(out_dims):
            if not used[i] and od == sd:
                result.append(i); used[i] = True; matched = True; break
        if not matched:
            for i in range(len(out_dims)):
                if not used[i]:
                    result.append(i); used[i] = True; break
    return result

def _find_expand_reassoc(A, C):
    """Find reassociation indices mapping collapsed dims A to expanded dims C."""
    ndA, ndC = len(A), len(C)
    def helper(ai, ci):
        if ai == ndA:
            return [] if ci == ndC else None
        max_size = ndC - ci - (ndA - ai - 1)
        for size in range(1, max_size + 1):
            grp = list(range(ci, ci + size))
            a_dim = A[ai]
            if isinstance(a_dim, int):
                prod = 1; has_dyn = False
                for g in grp:
                    d = C[g]
                    if not isinstance(d, int):
                        has_dyn = True; break
                    prod *= d
                if not has_dyn and prod == a_dim:
                    rest = helper(ai + 1, ci + size)
                    if rest is not None:
                        return [grp] + rest
        return None
    return helper(0, 0)

def _concat_axis(inputs, C):
    """Return the concatenation axis of inputs into output C."""
    nd = len(C)
    for d in range(nd):
        if all(isinstance(inp[d], int) for inp in inputs) and isinstance(C[d], int):
            if sum(inp[d] for inp in inputs) == C[d]:
                return d
    return nd - 1

# ---------------------------------------------------------------------------
# MLIR text helpers
# ---------------------------------------------------------------------------

def _fname(stem):
    return ("f_" + stem) if stem and stem[0].isdigit() else stem

def _build_module(fn_name, arg_pairs, body_lines, ret_dims, ret_dtype="f32"):
    """Wrap body_lines in: module @X { func.func @X(...) -> Y { ... } }"""
    arg_strs = ["%{}: {}".format(n, mt(d, dt))
                for n, d, dt in arg_pairs]
    ret_t   = mt(ret_dims, ret_dtype)
    header  = "  func.func @{}({}) -> {} {{".format(fn_name, ", ".join(arg_strs), ret_t)
    lines   = ["module @{} {{".format(fn_name), header]
    lines  += body_lines
    lines  += ["  }", "}"]
    return "\n".join(lines) + "\n"

def _cst_indices(n):
    return ["    %c{} = arith.constant {} : index".format(i, i) for i in range(n)]

def _load_dims(var, dims, dtype="f32"):
    t = mt(dims, dtype)
    return ["    %{v}_d{i} = memref.dim %{v}, %c{i} : {t}".format(v=var, i=i, t=t)
            for i in range(len(dims))]

def _open_for(depth, lvar, bound_expr):
    ind = "    " + "  " * depth
    return "{}scf.for %{} = %c0 to {} step %c1 {{".format(ind, lvar, bound_expr)

def _close_for(depth):
    return ("    " + "  " * depth) + "}"

def _ind(depth):
    return "    " + "  " * depth

def _assert_eq(va, vb, msg, eqvar):
    return [
        "    {} = arith.cmpi eq, {}, {} : index".format(eqvar, va, vb),
        "    cf.assert {}, \"{}\"".format(eqvar, msg),
    ]

def _idx_str(lvars):
    return "[" + ", ".join("%{}".format(v) for v in lvars) + "]"

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

# ---- relu ----

def gen_relu(stem, shapes):
    if not shapes or not is_static(shapes[0]):
        return None
    A = shapes[0]; nd = len(A); t = mt(A); fn = _fname(stem)
    body  = _cst_indices(nd) + _load_dims("input", A)
    body += ["    %out = memref.alloc() : {}".format(t)]
    lvars = ["ui{}".format(i) for i in range(nd)]
    for i in range(nd):
        body.append(_open_for(i, lvars[i], "%input_d{}".format(i)))
    ii  = _ind(nd); idx = _idx_str(lvars)
    body += [
        "{}%in_val  = memref.load %input{} : {}".format(ii, idx, t),
        "{}%zero_f  = arith.constant 0.0 : f32".format(ii),
        "{}%out_val = arith.maximumf %in_val, %zero_f : f32".format(ii),
        "{}memref.store %out_val, %out{} : {}".format(ii, idx, t),
    ]
    for i in range(nd - 1, -1, -1):
        body.append(_close_for(i))
    body += ["    return %out : {}".format(t)]
    return _build_module(fn, [("input", A, "f32")], body, A)

# ---- sigmoid ----

def gen_sigmoid(stem, shapes):
    if not shapes or not is_static(shapes[0]):
        return None
    A = shapes[0]; nd = len(A); t = mt(A); fn = _fname(stem)
    body  = _cst_indices(nd) + _load_dims("input", A)
    body += ["    %out = memref.alloc() : {}".format(t)]
    lvars = ["si{}".format(i) for i in range(nd)]
    for i in range(nd):
        body.append(_open_for(i, lvars[i], "%input_d{}".format(i)))
    ii = _ind(nd); idx = _idx_str(lvars)
    body += [
        "{}%xv    = memref.load %input{} : {}".format(ii, idx, t),
        "{}%one   = arith.constant 1.0 : f32".format(ii),
        "{}%negx  = arith.negf %xv : f32".format(ii),
        "{}%expv  = math.exp %negx : f32".format(ii),
        "{}%denom = arith.addf %one, %expv : f32".format(ii),
        "{}%out_val = arith.divf %one, %denom : f32".format(ii),
        "{}memref.store %out_val, %out{} : {}".format(ii, idx, t),
    ]
    for i in range(nd - 1, -1, -1):
        body.append(_close_for(i))
    body += ["    return %out : {}".format(t)]
    return _build_module(fn, [("input", A, "f32")], body, A)

# ---- gelu ----

def gen_gelu(stem, shapes):
    if not shapes or not is_static(shapes[0]):
        return None
    A = shapes[0]; nd = len(A); t = mt(A); fn = _fname(stem)
    body  = _cst_indices(nd) + _load_dims("input", A)
    body += ["    %out = memref.alloc() : {}".format(t)]
    lvars = ["gi{}".format(i) for i in range(nd)]
    for i in range(nd):
        body.append(_open_for(i, lvars[i], "%input_d{}".format(i)))
    ii = _ind(nd); idx = _idx_str(lvars)
    body += [
        "{}%xv      = memref.load %input{} : {}".format(ii, idx, t),
        "{}%half    = arith.constant 5.000000e-01 : f32".format(ii),
        "{}%one     = arith.constant 1.0 : f32".format(ii),
        "{}%sqrt2pi = arith.constant 7.978846e-01 : f32".format(ii),
        "{}%coeff   = arith.constant 4.471500e-02 : f32".format(ii),
        "{}%x2      = arith.mulf %xv, %xv : f32".format(ii),
        "{}%x3      = arith.mulf %x2, %xv : f32".format(ii),
        "{}%cx3     = arith.mulf %coeff, %x3 : f32".format(ii),
        "{}%inner   = arith.addf %xv, %cx3 : f32".format(ii),
        "{}%targ    = arith.mulf %sqrt2pi, %inner : f32".format(ii),
        "{}%tv      = math.tanh %targ : f32".format(ii),
        "{}%one_tv  = arith.addf %one, %tv : f32".format(ii),
        "{}%hx      = arith.mulf %half, %xv : f32".format(ii),
        "{}%out_val = arith.mulf %hx, %one_tv : f32".format(ii),
        "{}memref.store %out_val, %out{} : {}".format(ii, idx, t),
    ]
    for i in range(nd - 1, -1, -1):
        body.append(_close_for(i))
    body += ["    return %out : {}".format(t)]
    return _build_module(fn, [("input", A, "f32")], body, A)

# ---- elemwise_add ----

def gen_elemwise_add(stem, shapes):
    if len(shapes) < 2:
        return None
    A, B = shapes[0], shapes[1]
    C = shapes[2] if len(shapes) >= 3 else A
    if not is_static(A) or not is_static(B) or not is_static(C):
        return None
    nd   = len(A)
    t_a  = mt(A); t_b = mt(B); t_out = mt(C); fn = _fname(stem)
    body = _cst_indices(nd) + _load_dims("in0", A) + _load_dims("in1", B)

    # Shape assertions: for matching non-broadcast dims
    if B == A:
        # Same shape: every dim must match
        for i in range(nd):
            body += _assert_eq("%in0_d{}".format(i), "%in1_d{}".format(i),
                               "in0.dim({})==in1.dim({})".format(i, i),
                               "%_aeq{}".format(i))
    else:
        # Broadcast: assert non-unit B dims equal the corresponding A dim
        bcast = _find_bcast_dims(B, A)
        for k in range(len(B)):
            if B[k] > 1:
                body += _assert_eq("%in1_d{}".format(k), "%in0_d{}".format(bcast[k]),
                                   "in1.dim({})==in0.dim({})".format(k, bcast[k]),
                                   "%_baeq{}".format(k))

    body += ["    %out = memref.alloc() : {}".format(t_out)]
    lvars = ["ae{}".format(i) for i in range(nd)]
    for i in range(nd):
        body.append(_open_for(i, lvars[i], "%in0_d{}".format(i)))
    ii     = _ind(nd)
    out_id = _idx_str(lvars)
    body  += ["{}%v0 = memref.load %in0{} : {}".format(ii, out_id, t_a)]

    # Load in1 with broadcast-aware indexing
    if B == A:
        body.append("{}%v1 = memref.load %in1{} : {}".format(ii, out_id, t_b))
    else:
        bcast = _find_bcast_dims(B, A)
        in1_ids = []
        for k in range(len(B)):
            if B[k] == 1:
                in1_ids.append("%c0")
            else:
                in1_ids.append("%{}".format(lvars[bcast[k]]))
        body.append("{}%v1 = memref.load %in1{} : {}".format(
            ii, "[" + ", ".join(in1_ids) + "]", t_b))

    body += [
        "{}%out_val = arith.addf %v0, %v1 : f32".format(ii),
        "{}memref.store %out_val, %out{} : {}".format(ii, out_id, t_out),
    ]
    for i in range(nd - 1, -1, -1):
        body.append(_close_for(i))
    body += ["    return %out : {}".format(t_out)]
    return _build_module(fn, [("in0", A, "f32"), ("in1", B, "f32")], body, C)

# ---- softmax ----

def gen_softmax(stem, shapes):
    if not shapes or not is_static(shapes[0]):
        return None
    A = shapes[0]; nd = len(A); t = mt(A); fn = _fname(stem)
    rd = nd - 1   # reduce dimension (last axis)
    body = _cst_indices(nd) + _load_dims("input", A)
    body += ["    %out = memref.alloc() : {}".format(t)]

    lvars_outer = ["so{}".format(i) for i in range(nd - 1)]
    for i in range(nd - 1):
        body.append(_open_for(i, lvars_outer[i], "%input_d{}".format(i)))
    oi = _ind(nd - 1)
    out_prefix = ", ".join("%{}".format(v) for v in lvars_outer)

    def _load_at(inner_var):
        if out_prefix:
            return "[{}, %{}]".format(out_prefix, inner_var)
        return "[%{}]".format(inner_var)

    def _store_at(inner_var):
        if out_prefix:
            return "[{}, %{}]".format(out_prefix, inner_var)
        return "[%{}]".format(inner_var)

    # 1. Find max
    body += [
        "{}%neg_inf = arith.constant -3.4028234663852886e+38 : f32".format(oi),
        "{}%max_val = scf.for %sk = %c0 to %input_d{rd} step %c1"
        " iter_args(%mx = %neg_inf) -> (f32) {{".format(oi, rd=rd),
        "{}  %pv1 = memref.load %input{} : {}".format(oi, _load_at("sk"), t),
        "{}  %gt  = arith.cmpf ogt, %pv1, %mx : f32".format(oi),
        "{}  %nx  = arith.select %gt, %pv1, %mx : f32".format(oi),
        "{}  scf.yield %nx : f32".format(oi),
        "{}}}".format(oi),
        # 2. Sum of exp(x - max)
        "{}%zero_f  = arith.constant 0.0 : f32".format(oi),
        "{}%sum_val = scf.for %sk2 = %c0 to %input_d{rd} step %c1"
        " iter_args(%sm = %zero_f) -> (f32) {{".format(oi, rd=rd),
        "{}  %pv2     = memref.load %input{} : {}".format(oi, _load_at("sk2"), t),
        "{}  %shifted = arith.subf %pv2, %max_val : f32".format(oi),
        "{}  %expv    = math.exp %shifted : f32".format(oi),
        "{}  %ns2     = arith.addf %sm, %expv : f32".format(oi),
        "{}  scf.yield %ns2 : f32".format(oi),
        "{}}}".format(oi),
        # 3. Normalize and write output
        "{}scf.for %sk3 = %c0 to %input_d{rd} step %c1 {{".format(oi, rd=rd),
        "{}  %pv3  = memref.load %input{} : {}".format(oi, _load_at("sk3"), t),
        "{}  %sh3  = arith.subf %pv3, %max_val : f32".format(oi),
        "{}  %ex3  = math.exp %sh3 : f32".format(oi),
        "{}  %ov   = arith.divf %ex3, %sum_val : f32".format(oi),
        "{}  memref.store %ov, %out{} : {}".format(oi, _store_at("sk3"), t),
        "{}}}".format(oi),
    ]
    for i in range(nd - 2, -1, -1):
        body.append(_close_for(i))
    body += ["    return %out : {}".format(t)]
    return _build_module(fn, [("input", A, "f32")], body, A)

# ---- matmul ----

def gen_matmul(stem, shapes):
    if len(shapes) < 3:
        return None
    A, B, C = shapes[0], shapes[1], shapes[2]
    if not is_static(A) or not is_static(B) or not is_static(C):
        return None
    ndA = len(A); ndB = len(B); fn = _fname(stem)
    t_a = mt(A); t_b = mt(B); t_c = mt(C)

    body = _cst_indices(max(ndA, ndB, len(C)))
    body += _load_dims("lhs", A) + _load_dims("rhs", B)
    body += ["    %out = memref.alloc() : {}".format(t_c)]

    if ndA == 2 and ndB == 2:
        # [M, K] x [K, N] → [M, N]
        body += _assert_eq("%lhs_d1", "%rhs_d0", "lhs.dim(1)==rhs.dim(0)", "%_meq0")
        body += [
            _open_for(0, "m", "%lhs_d0"),
            _open_for(1, "n", "%rhs_d1"),
            "{}%zero_f = arith.constant 0.0 : f32".format(_ind(2)),
            "{}%acc = scf.for %k = %c0 to %lhs_d1 step %c1"
            " iter_args(%s = %zero_f) -> (f32) {{".format(_ind(2)),
            "{}  %a  = memref.load %lhs[%m, %k] : {}".format(_ind(2), t_a),
            "{}  %b  = memref.load %rhs[%k, %n] : {}".format(_ind(2), t_b),
            "{}  %p  = arith.mulf %a, %b : f32".format(_ind(2)),
            "{}  %ns = arith.addf %s, %p : f32".format(_ind(2)),
            "{}  scf.yield %ns : f32".format(_ind(2)),
            "{}}}".format(_ind(2)),
            "{}memref.store %acc, %out[%m, %n] : {}".format(_ind(2), t_c),
            _close_for(1), _close_for(0),
        ]

    elif ndA == 3 and ndB == 2:
        # [B, M, K] x [K, N] → [B, M, N]
        body += _assert_eq("%lhs_d2", "%rhs_d0", "lhs.dim(2)==rhs.dim(0)", "%_meq0")
        body += [
            _open_for(0, "bs", "%lhs_d0"),
            _open_for(1, "m",  "%lhs_d1"),
            _open_for(2, "n",  "%rhs_d1"),
            "{}%zero_f = arith.constant 0.0 : f32".format(_ind(3)),
            "{}%acc = scf.for %k = %c0 to %lhs_d2 step %c1"
            " iter_args(%s = %zero_f) -> (f32) {{".format(_ind(3)),
            "{}  %a  = memref.load %lhs[%bs, %m, %k] : {}".format(_ind(3), t_a),
            "{}  %b  = memref.load %rhs[%k, %n] : {}".format(_ind(3), t_b),
            "{}  %p  = arith.mulf %a, %b : f32".format(_ind(3)),
            "{}  %ns = arith.addf %s, %p : f32".format(_ind(3)),
            "{}  scf.yield %ns : f32".format(_ind(3)),
            "{}}}".format(_ind(3)),
            "{}memref.store %acc, %out[%bs, %m, %n] : {}".format(_ind(3), t_c),
            _close_for(2), _close_for(1), _close_for(0),
        ]

    elif ndA == 3 and ndB == 3:
        # [B, M, K] x [B, K, N] → [B, M, N]
        body += _assert_eq("%lhs_d0", "%rhs_d0", "lhs.dim(0)==rhs.dim(0)", "%_meq0")
        body += _assert_eq("%lhs_d2", "%rhs_d1", "lhs.dim(2)==rhs.dim(1)", "%_meq1")
        body += [
            _open_for(0, "bs", "%lhs_d0"),
            _open_for(1, "m",  "%lhs_d1"),
            _open_for(2, "n",  "%rhs_d2"),
            "{}%zero_f = arith.constant 0.0 : f32".format(_ind(3)),
            "{}%acc = scf.for %k = %c0 to %lhs_d2 step %c1"
            " iter_args(%s = %zero_f) -> (f32) {{".format(_ind(3)),
            "{}  %a  = memref.load %lhs[%bs, %m, %k] : {}".format(_ind(3), t_a),
            "{}  %b  = memref.load %rhs[%bs, %k, %n] : {}".format(_ind(3), t_b),
            "{}  %p  = arith.mulf %a, %b : f32".format(_ind(3)),
            "{}  %ns = arith.addf %s, %p : f32".format(_ind(3)),
            "{}  scf.yield %ns : f32".format(_ind(3)),
            "{}}}".format(_ind(3)),
            "{}memref.store %acc, %out[%bs, %m, %n] : {}".format(_ind(3), t_c),
            _close_for(2), _close_for(1), _close_for(0),
        ]

    elif ndA == 4:
        # [B1, B2, M, K] x [B1, B2, K, N] → [B1, B2, M, N]
        body += _assert_eq("%lhs_d0", "%rhs_d0", "lhs.dim(0)==rhs.dim(0)", "%_meq0")
        body += _assert_eq("%lhs_d1", "%rhs_d1", "lhs.dim(1)==rhs.dim(1)", "%_meq1")
        body += _assert_eq("%lhs_d3", "%rhs_d2", "lhs.dim(3)==rhs.dim(2)", "%_meq2")
        body += [
            _open_for(0, "b1", "%lhs_d0"),
            _open_for(1, "b2", "%lhs_d1"),
            _open_for(2, "m",  "%lhs_d2"),
            _open_for(3, "n",  "%rhs_d3"),
            "{}%zero_f = arith.constant 0.0 : f32".format(_ind(4)),
            "{}%acc = scf.for %k = %c0 to %lhs_d3 step %c1"
            " iter_args(%s = %zero_f) -> (f32) {{".format(_ind(4)),
            "{}  %a  = memref.load %lhs[%b1, %b2, %m, %k] : {}".format(_ind(4), t_a),
            "{}  %b  = memref.load %rhs[%b1, %b2, %k, %n] : {}".format(_ind(4), t_b),
            "{}  %p  = arith.mulf %a, %b : f32".format(_ind(4)),
            "{}  %ns = arith.addf %s, %p : f32".format(_ind(4)),
            "{}  scf.yield %ns : f32".format(_ind(4)),
            "{}}}".format(_ind(4)),
            "{}memref.store %acc, %out[%b1, %b2, %m, %n] : {}".format(_ind(4), t_c),
            _close_for(3), _close_for(2), _close_for(1), _close_for(0),
        ]
    else:
        return None

    body += ["    return %out : {}".format(t_c)]
    return _build_module(fn, [("lhs", A, "f32"), ("rhs", B, "f32")], body, C)

# ---- transpose ----

def gen_transpose(stem, shapes):
    if len(shapes) < 2:
        return None
    A, C = shapes[0], shapes[1]
    if not is_static(A) or not is_static(C):
        return None
    ndA = len(A); t_a = mt(A); t_c = mt(C); fn = _fname(stem)
    # Find permutation
    perm = None
    for p in permutations(range(ndA)):
        if [A[i] for i in p] == C:
            perm = list(p); break
    if perm is None:
        return None
    body = _cst_indices(ndA) + _load_dims("input", A)
    body += ["    %out = memref.alloc() : {}".format(t_c)]

    # Loop over output shape (equivalent to looping over input shape with perm)
    out_lvars = ["pi{}".format(i) for i in range(ndA)]
    # inv_perm: for each output axis, which input axis it came from (= perm[i])
    # Loop var pi_i traverses output axis i, which is input axis perm[i]
    for i in range(ndA):
        # bound = input.dim(perm[i])
        body.append(_open_for(i, out_lvars[i], "%input_d{}".format(perm[i])))
    ii = _ind(ndA)
    # input index: for each input axis j, find which out loop var gives it
    # out loop var pi_i traverses input axis perm[i]
    # So input[j] is indexed by out loop var pi_{inv_perm[j]}
    inv_perm = [0] * ndA
    for i, p in enumerate(perm):
        inv_perm[p] = i
    in_ids  = "[" + ", ".join("%{}".format(out_lvars[inv_perm[j]]) for j in range(ndA)) + "]"
    out_ids = "[" + ", ".join("%{}".format(out_lvars[i]) for i in range(ndA)) + "]"
    body += [
        "{}%tv    = memref.load %input{} : {}".format(ii, in_ids, t_a),
        "{}memref.store %tv, %out{} : {}".format(ii, out_ids, t_c),
    ]
    for i in range(ndA - 1, -1, -1):
        body.append(_close_for(i))
    body += ["    return %out : {}".format(t_c)]
    return _build_module(fn, [("input", A, "f32")], body, C)

# ---- reshape ----

def gen_reshape(stem, shapes):
    if len(shapes) < 2:
        return None
    A, C = shapes[0], shapes[1]
    if not is_static(A) or not is_static(C):
        return None
    t_a = mt(A); t_c = mt(C); fn = _fname(stem)
    body = ["    // reshape: no shape assertions required (element count preserved by type)"]

    ndA, ndC = len(A), len(C)
    # Try expand_shape (A has fewer dims than C)
    if ndC > ndA:
        reassoc = _find_expand_reassoc(A, C)
        if reassoc is not None:
            reassoc_str = "[{}]".format(", ".join("[{}]".format(", ".join(str(x) for x in g))
                                                   for g in reassoc))
            out_shape_str = "[{}]".format(", ".join(str(d) for d in C))
            body += [
                "    %out = memref.expand_shape %input {} output_shape {} : {} into {}".format(
                    reassoc_str, out_shape_str, t_a, t_c),
                "    return %out : {}".format(t_c),
            ]
            return _build_module(fn, [("input", A, "f32")], body, C)
    # Try collapse_shape (A has more dims than C)
    if ndA > ndC:
        reassoc = _find_expand_reassoc(C, A)    # reverse: collapse A into C
        if reassoc is not None:
            # reassoc built for C→A; need groups for the collapse direction
            # For collapse: reassoc for A→C groups input dims into each output dim
            reassoc_str = "[{}]".format(", ".join("[{}]".format(", ".join(str(x) for x in g))
                                                   for g in reassoc))
            body += [
                "    %out = memref.collapse_shape %input {} : {} into {}".format(
                    reassoc_str, t_a, t_c),
                "    return %out : {}".format(t_c),
            ]
            return _build_module(fn, [("input", A, "f32")], body, C)

    # Fallback: flat-loop copy (generic reshape)
    total = 1
    for d in A:
        total *= d
    body_flat = _cst_indices(max(ndA, ndC))
    out_shape_str = "[{}]".format(", ".join(str(d) for d in C))
    expand_reassoc = "[{}]".format(", ".join(str(i) for i in range(ndC)))
    body_flat += [
        "    %flat_in   = memref.collapse_shape %input [{}] : {} into memref<{}xf32>".format(
            "[{}]".format(", ".join(str(i) for i in range(ndA))), t_a, total),
        "    %flat_out  = memref.alloc() : memref<{}xf32>".format(total),
        "    %flat_n    = arith.constant {} : index".format(total),
        _open_for(0, "fi", "%flat_n"),
        "{}  %fv = memref.load %flat_in[%fi] : memref<{}xf32>".format(_ind(0), total),
        "{}  memref.store %fv, %flat_out[%fi] : memref<{}xf32>".format(_ind(0), total),
        _close_for(0),
        "    %out = memref.expand_shape %flat_out [{}] output_shape {} : memref<{}xf32> into {}".format(
            expand_reassoc, out_shape_str, total, t_c),
        "    return %out : {}".format(t_c),
    ]
    return _build_module(fn, [("input", A, "f32")], body_flat, C)

# ---- concat ----

def gen_concat(stem, shapes):
    if len(shapes) < 3:
        return None
    inputs = shapes[:-1]; C = shapes[-1]
    if not all(is_static(inp) for inp in inputs) or not is_static(C):
        return None
    n_inputs = len(inputs); nd = len(C)
    axis = _concat_axis(inputs, C)
    t_out = mt(C); fn = _fname(stem)

    body = _cst_indices(nd)
    # Load dims for all inputs
    for k, inp in enumerate(inputs):
        body += _load_dims("in{}".format(k), inp)

    # Shape assertions: non-concat axes of each in_k must match in0
    for k in range(1, n_inputs):
        for d in range(nd):
            if d == axis:
                continue
            body += _assert_eq(
                "%in0_d{}".format(d), "%in{}_d{}".format(k, d),
                "in0.dim({d})==in{k}.dim({d})".format(d=d, k=k),
                "%_cceq{k}{d}".format(k=k, d=d),
            )

    body += ["    %out = memref.alloc() : {}".format(t_out)]

    # Compute cumulative offsets along the concat axis
    # offset[0] = 0, offset[k] = offset[k-1] + in_{k-1}.dim(axis)
    offset_vars = ["%c0"]
    for k in range(1, n_inputs):
        ov = "%off_{}".format(k)
        prev = offset_vars[-1]
        prev_dim = "%in{}_d{}".format(k - 1, axis)
        body.append("    {} = arith.addi {}, {} : index".format(ov, prev, prev_dim))
        offset_vars.append(ov)

    # One copy block per input
    inp_lvars_all = []
    for k, inp in enumerate(inputs):
        t_ink = mt(inp); prefix = "cc{}_".format(k)
        lvars  = ["{}d{}".format(prefix, i) for i in range(nd)]
        inp_lvars_all.append(lvars)
        for i in range(nd):
            body.append(_open_for(i, lvars[i], "%in{}_d{}".format(k, i)))
        ii = _ind(nd)
        in_id = _idx_str(lvars)
        # Build output index: on axis, add the cumulative offset
        out_ids = []
        for d in range(nd):
            if d == axis:
                out_ids.append("arith.addi %{}, {} : index".format(lvars[d], offset_vars[k]))
            else:
                out_ids.append("%{}".format(lvars[d]))
        # We need SSA values for the offset-added index
        if nd > 0 and any("arith.addi" in x for x in out_ids):
            shift_var = "%cc{}_ax_out".format(k)
            ax_lvar = "%{}".format(lvars[axis])
            body.append("{}{}  = arith.addi {}, {} : index".format(
                ii, shift_var, ax_lvar, offset_vars[k]))
            final_out_ids = []
            for d in range(nd):
                if d == axis:
                    final_out_ids.append(shift_var)
                else:
                    final_out_ids.append("%{}".format(lvars[d]))
            out_idx = "[" + ", ".join(final_out_ids) + "]"
        else:
            out_idx = "[" + ", ".join(out_ids) + "]"
        body += [
            "{}%ccv{} = memref.load %in{}{} : {}".format(ii, k, k, in_id, t_ink),
            "{}memref.store %ccv{}, %out{} : {}".format(ii, k, out_idx, t_out),
        ]
        for i in range(nd - 1, -1, -1):
            body.append(_close_for(i))

    body += ["    return %out : {}".format(t_out)]
    arg_pairs = [("in{}".format(k), inp, "f32") for k, inp in enumerate(inputs)]
    return _build_module(fn, arg_pairs, body, C)

# ---- batch_norm ----

def gen_batch_norm(stem, shapes):
    if len(shapes) < 3:
        return None
    inp  = shapes[0]
    gamma_dims = shapes[1]
    beta_dims  = shapes[2] if len(shapes) >= 4 else shapes[1]
    if not is_static(inp) or not is_static(gamma_dims):
        return None
    nd = len(inp)
    # Find feature axis (the axis whose size equals gamma_dims[0])
    ch_axis = 1
    for ai, av in enumerate(inp):
        if av == gamma_dims[0]:
            ch_axis = ai; break
    t_in = mt(inp); t_g = mt(gamma_dims); t_b = mt(beta_dims)
    fn = _fname(stem)

    body = _cst_indices(nd) + _load_dims("input", inp)
    body += _load_dims("gamma", gamma_dims) + _load_dims("beta", beta_dims)
    # Shape assertions
    body += _assert_eq("%gamma_d0", "%input_d{}".format(ch_axis),
                       "gamma.dim(0)==input.dim({})".format(ch_axis), "%_bneq0")
    body += _assert_eq("%beta_d0", "%input_d{}".format(ch_axis),
                       "beta.dim(0)==input.dim({})".format(ch_axis),  "%_bneq1")

    body += ["    %out = memref.alloc() : {}".format(t_in)]
    lvars = ["bn{}".format(i) for i in range(nd)]
    for i in range(nd):
        body.append(_open_for(i, lvars[i], "%input_d{}".format(i)))
    ii  = _ind(nd)
    idx = _idx_str(lvars)
    body += [
        "{}%xv  = memref.load %input{} : {}".format(ii, idx, t_in),
        "{}%gv  = memref.load %gamma[%{}] : {}".format(ii, lvars[ch_axis], t_g),
        "{}%bv  = memref.load %beta[%{}]  : {}".format(ii, lvars[ch_axis], t_b),
        "{}%scl = arith.mulf %xv, %gv : f32".format(ii),
        "{}%res = arith.addf %scl, %bv : f32".format(ii),
        "{}memref.store %res, %out{} : {}".format(ii, idx, t_in),
    ]
    for i in range(nd - 1, -1, -1):
        body.append(_close_for(i))
    body += ["    return %out : {}".format(t_in)]
    return _build_module(fn, [("input", inp, "f32"), ("gamma", gamma_dims, "f32"),
                               ("beta", beta_dims, "f32")], body, inp)

# ---- layer_normalization ----

def gen_layer_norm(stem, shapes):
    if len(shapes) < 2:
        return None
    inp        = shapes[0]
    gamma_dims = shapes[1]
    beta_dims  = shapes[2] if len(shapes) >= 3 else gamma_dims
    if not is_static(inp) or not is_static(gamma_dims):
        return None
    nd         = len(inp)
    norm_rank  = len(gamma_dims)
    batch_rank = nd - norm_rank
    norm_size  = 1
    for d in inp[batch_rank:]:
        norm_size *= d
    t_in = mt(inp); t_g = mt(gamma_dims); t_bt = mt(beta_dims)
    fn = _fname(stem)

    body = _cst_indices(nd) + _load_dims("input", inp)
    body += _load_dims("gamma", gamma_dims) + _load_dims("beta", beta_dims)
    # Shape assertions: gamma/beta last-dim == input last norm dim
    norm_axis = batch_rank   # first norm axis in input
    body += _assert_eq("%gamma_d0", "%input_d{}".format(norm_axis),
                       "gamma.dim(0)==input.dim({})".format(norm_axis), "%_lneq0")
    body += _assert_eq("%beta_d0", "%input_d{}".format(norm_axis),
                       "beta.dim(0)==input.dim({})".format(norm_axis),  "%_lneq1")

    body += ["    %out = memref.alloc() : {}".format(t_in)]
    batch_lvars = ["ln_b{}".format(i) for i in range(batch_rank)]
    norm_lvars  = ["ln_n{}".format(i) for i in range(norm_rank)]
    for i in range(batch_rank):
        body.append(_open_for(i, batch_lvars[i], "%input_d{}".format(i)))
    oi = _ind(batch_rank)
    batch_idx = ", ".join("%{}".format(v) for v in batch_lvars)

    def _full_idx(norm_vars):
        """Build '[batch..., norm...]' index string."""
        parts = ["%{}".format(v) for v in batch_lvars] + ["%{}".format(v) for v in norm_vars]
        return "[" + ", ".join(parts) + "]"

    def _gamma_idx(norm_vars):
        return "[" + ", ".join("%{}".format(v) for v in norm_vars) + "]"

    nsz_str = "{}.0".format(norm_size)
    body += [
        "{}%zero_f = arith.constant 0.0 : f32".format(oi),
        "{}%eps    = arith.constant 1.0e-05 : f32".format(oi),
        "{}%nsz    = arith.constant {} : f32".format(oi, nsz_str),
    ]

    # ---- Pass 1: compute sum via nested iter_args loops ----
    # Outer norm loops use iter_args chaining (inner yields into outer)
    # e.g. norm_rank=2:
    #   %ln_sum = scf.for %ln_n0 ... iter_args(%sa0 = %zero_f) -> (f32) {
    #     %inner = scf.for %ln_n1 ... iter_args(%sa1 = %sa0) -> (f32) {
    #       %lnv = load ...; %ns = addf %sa1, %lnv; yield %ns
    #     }
    #     yield %inner
    #   }
    sum_acc_names  = ["ln_sa{}".format(i) for i in range(norm_rank)]
    sumsq_acc_names = ["ln_sq{}".format(i) for i in range(norm_rank)]
    ii_inner = oi + "  " * norm_rank
    for ri in range(norm_rank):
        ind = oi + "  " * ri
        acc = sum_acc_names[ri]
        prev = "%zero_f" if ri == 0 else "%{}".format(sum_acc_names[ri - 1])
        if ri == 0:
            body.append(
                "{}%ln_sum = scf.for %{} = %c0 to %input_d{} step %c1"
                " iter_args(%{} = {}) -> (f32) {{".format(
                    ind, norm_lvars[ri], norm_axis + ri, acc, prev))
        else:
            body.append(
                "{}  %ln_sum{} = scf.for %{} = %c0 to %input_d{} step %c1"
                " iter_args(%{} = {}) -> (f32) {{".format(
                    ind, ri, norm_lvars[ri], norm_axis + ri, acc, prev))
    body += [
        "{}%lnv1 = memref.load %input{} : {}".format(
            ii_inner, _full_idx(norm_lvars), t_in),
        "{}%ns1  = arith.addf %{}, %lnv1 : f32".format(
            ii_inner, sum_acc_names[-1]),
        "{}scf.yield %ns1 : f32".format(ii_inner),
    ]
    for ri in range(norm_rank - 1, -1, -1):
        ind = oi + "  " * ri
        if ri < norm_rank - 1:
            body.append("{}  scf.yield %ln_sum{} : f32".format(ind, ri + 1))
        body.append("{}}}".format(ind + "  "))
    body.append("{}%ln_mean = arith.divf %ln_sum, %nsz : f32".format(oi))

    # ---- Pass 2: compute variance sum ----
    for ri in range(norm_rank):
        ind = oi + "  " * ri
        acc = sumsq_acc_names[ri]
        prev = "%zero_f" if ri == 0 else "%{}".format(sumsq_acc_names[ri - 1])
        if ri == 0:
            body.append(
                "{}%ln_sumsq = scf.for %{}v = %c0 to %input_d{} step %c1"
                " iter_args(%{} = {}) -> (f32) {{".format(
                    ind, norm_lvars[ri], norm_axis + ri, acc, prev))
        else:
            body.append(
                "{}  %ln_sumsq{} = scf.for %{}v = %c0 to %input_d{} step %c1"
                " iter_args(%{} = {}) -> (f32) {{".format(
                    ind, ri, norm_lvars[ri], norm_axis + ri, acc, prev))
    norm_vars_v = [v + "v" for v in norm_lvars]
    body += [
        "{}%lnv2 = memref.load %input{} : {}".format(
            ii_inner, _full_idx(norm_vars_v), t_in),
        "{}%lnd2 = arith.subf %lnv2, %ln_mean : f32".format(ii_inner),
        "{}%sq   = arith.mulf %lnd2, %lnd2 : f32".format(ii_inner),
        "{}%ns2  = arith.addf %{}, %sq : f32".format(ii_inner, sumsq_acc_names[-1]),
        "{}scf.yield %ns2 : f32".format(ii_inner),
    ]
    for ri in range(norm_rank - 1, -1, -1):
        ind = oi + "  " * ri
        if ri < norm_rank - 1:
            body.append("{}  scf.yield %ln_sumsq{} : f32".format(ind, ri + 1))
        body.append("{}}}".format(ind + "  "))
    body += [
        "{}%ln_var    = arith.divf %ln_sumsq, %nsz : f32".format(oi),
        "{}%ln_veps   = arith.addf %ln_var, %eps : f32".format(oi),
        "{}%ln_invstd = math.rsqrt %ln_veps : f32".format(oi),
    ]

    # ---- Pass 3: normalize, scale, shift (no iter_args — memref stores) ----
    norm_vars_w = ["ln_w{}".format(i) for i in range(norm_rank)]
    for ri in range(norm_rank):
        body.append(_open_for(batch_rank + ri, norm_vars_w[ri],
                              "%input_d{}".format(norm_axis + ri)))
    ii3 = _ind(batch_rank + norm_rank)
    body += [
        "{}%lnv3    = memref.load %input{} : {}".format(
            ii3, _full_idx(norm_vars_w), t_in),
        "{}%lnd3    = arith.subf %lnv3, %ln_mean : f32".format(ii3),
        "{}%ln_norm = arith.mulf %lnd3, %ln_invstd : f32".format(ii3),
        "{}%gv      = memref.load %gamma{} : {}".format(
            ii3, _gamma_idx(norm_vars_w), t_g),
        "{}%bv      = memref.load %beta{}  : {}".format(
            ii3, _gamma_idx(norm_vars_w), t_bt),
        "{}%scaled  = arith.mulf %ln_norm, %gv : f32".format(ii3),
        "{}%ln_res  = arith.addf %scaled, %bv : f32".format(ii3),
        "{}memref.store %ln_res, %out{} : {}".format(
            ii3, _full_idx(norm_vars_w), t_in),
    ]
    for ri in range(norm_rank - 1, -1, -1):
        body.append(_close_for(batch_rank + ri))
    for i in range(batch_rank - 1, -1, -1):
        body.append(_close_for(i))
    body += ["    return %out : {}".format(t_in)]
    return _build_module(fn, [("input", inp, "f32"), ("gamma", gamma_dims, "f32"),
                               ("beta", beta_dims, "f32")], body, inp)

# ---- conv2d ----

def gen_conv2d(stem, shapes):
    if len(shapes) < 3:
        return None
    inp, filt, out = shapes[0], shapes[1], shapes[2]
    if not is_static(inp) or not is_static(filt) or not is_static(out):
        return None
    if len(inp) != 4 or len(filt) != 4:
        return None
    kH = filt[2]; kW = filt[3]
    strides = max(1, inp[2] // out[2]) if out[2] > 0 else 1
    t_in = mt(inp); t_f = mt(filt); t_out = mt(out); fn = _fname(stem)

    body = _cst_indices(4) + _load_dims("input", inp) + _load_dims("filter", filt)
    # Assertion: input channels == filter channels
    body += _assert_eq("%input_d1", "%filter_d1",
                       "input.dim(1)==filter.dim(1)", "%_cveq0")

    body += ["    %out = memref.alloc() : {}".format(t_out)]
    kH_c = "arith.constant {} : index".format(kH)
    kW_c = "arith.constant {} : index".format(kW)
    str_c = "arith.constant {} : index".format(strides)
    body += [
        "    %kH_sz  = arith.constant {} : index".format(kH),
        "    %kW_sz  = arith.constant {} : index".format(kW),
        "    %stride = arith.constant {} : index".format(strides),
    ]
    body += [
        _open_for(0, "n",  "%input_d0"),
        _open_for(1, "f_", "%filter_d0"),
        _open_for(2, "oh", "%input_d2"),   # output H (simplified: same as input H for static)
        _open_for(3, "ow", "%input_d3"),   # output W
        "{}%zero_f = arith.constant 0.0 : f32".format(_ind(4)),
        "{}%acc = scf.for %c = %c0 to %input_d1 step %c1"
        " iter_args(%s_c = %zero_f) -> (f32) {{".format(_ind(4)),
        "{}  %acc2 = scf.for %kh = %c0 to %kH_sz step %c1"
        " iter_args(%s_kh = %s_c) -> (f32) {{".format(_ind(4)),
        "{}    %acc3 = scf.for %kw = %c0 to %kW_sz step %c1"
        " iter_args(%s_kw = %s_kh) -> (f32) {{".format(_ind(4)),
        "{}      %s_oh = arith.muli %stride, %oh : index".format(_ind(4)),
        "{}      %ih   = arith.addi %s_oh, %kh : index".format(_ind(4)),
        "{}      %s_ow = arith.muli %stride, %ow : index".format(_ind(4)),
        "{}      %iw   = arith.addi %s_ow, %kw : index".format(_ind(4)),
        "{}      %av   = memref.load %input[%n, %c, %ih, %iw] : {}".format(_ind(4), t_in),
        "{}      %bv   = memref.load %filter[%f_, %c, %kh, %kw] : {}".format(_ind(4), t_f),
        "{}      %pv   = arith.mulf %av, %bv : f32".format(_ind(4)),
        "{}      %ns   = arith.addf %s_kw, %pv : f32".format(_ind(4)),
        "{}      scf.yield %ns : f32".format(_ind(4)),
        "{}    }}".format(_ind(4)),
        "{}    scf.yield %acc3 : f32".format(_ind(4)),
        "{}  }}".format(_ind(4)),
        "{}  scf.yield %acc2 : f32".format(_ind(4)),
        "{}}}".format(_ind(4)),
        "{}memref.store %acc, %out[%n, %f_, %oh, %ow] : {}".format(_ind(4), t_out),
        _close_for(3), _close_for(2), _close_for(1), _close_for(0),
    ]
    body += ["    return %out : {}".format(t_out)]
    return _build_module(fn, [("input", inp, "f32"), ("filter", filt, "f32")], body, out)

# ---- max_pool2d ----

def gen_max_pool2d(stem, shapes):
    if len(shapes) < 2:
        return None
    inp, out_shape = shapes[0], shapes[1]
    if not is_static(inp) or not is_static(out_shape):
        return None
    if len(inp) < 4 or len(out_shape) < 4:
        return None
    strides = max(1, inp[2] // out_shape[2]) if out_shape[2] > 0 else 2
    kH = kW = strides
    t_in  = mt(inp); t_out = mt(out_shape); fn = _fname(stem)

    body = _cst_indices(4) + _load_dims("input", inp)
    body += ["    %out = memref.alloc() : {}".format(t_out)]
    body += [
        "    %kH_sz  = arith.constant {} : index".format(kH),
        "    %kW_sz  = arith.constant {} : index".format(kW),
        "    %stride = arith.constant {} : index".format(strides),
        "    %out_d2 = arith.constant {} : index".format(out_shape[2]),
        "    %out_d3 = arith.constant {} : index".format(out_shape[3]),
    ]
    body += [
        _open_for(0, "n",  "%input_d0"),
        _open_for(1, "c",  "%input_d1"),
        _open_for(2, "oh", "%out_d2"),
        _open_for(3, "ow", "%out_d3"),
        "{}%neg_inf = arith.constant -3.4028234663852886e+38 : f32".format(_ind(4)),
        "{}%max_v = scf.for %kh = %c0 to %kH_sz step %c1"
        " iter_args(%mx_h = %neg_inf) -> (f32) {{".format(_ind(4)),
        "{}  %max_v2 = scf.for %kw = %c0 to %kW_sz step %c1"
        " iter_args(%mx_w = %mx_h) -> (f32) {{".format(_ind(4)),
        "{}    %s_oh = arith.muli %stride, %oh : index".format(_ind(4)),
        "{}    %ih   = arith.addi %s_oh, %kh : index".format(_ind(4)),
        "{}    %s_ow = arith.muli %stride, %ow : index".format(_ind(4)),
        "{}    %iw   = arith.addi %s_ow, %kw : index".format(_ind(4)),
        "{}    %pv   = memref.load %input[%n, %c, %ih, %iw] : {}".format(_ind(4), t_in),
        "{}    %gt   = arith.cmpf ogt, %pv, %mx_w : f32".format(_ind(4)),
        "{}    %nx   = arith.select %gt, %pv, %mx_w : f32".format(_ind(4)),
        "{}    scf.yield %nx : f32".format(_ind(4)),
        "{}  }}".format(_ind(4)),
        "{}  scf.yield %max_v2 : f32".format(_ind(4)),
        "{}}}".format(_ind(4)),
        "{}memref.store %max_v, %out[%n, %c, %oh, %ow] : {}".format(_ind(4), t_out),
        _close_for(3), _close_for(2), _close_for(1), _close_for(0),
    ]
    body += ["    return %out : {}".format(t_out)]
    return _build_module(fn, [("input", inp, "f32")], body, out_shape)

# ---- reduce_mean ----

def gen_reduce_mean(stem, shapes):
    if len(shapes) < 2:
        return None
    inp = shapes[0]; out_declared = shapes[-1]
    if not is_static(inp):
        return None
    ndA = len(inp)
    # Determine reduce dims: axes in inp not present in out
    used = [False] * ndA
    for od in out_declared:
        for i in range(ndA):
            if not used[i] and isinstance(od, int) and inp[i] == od:
                used[i] = True; break
    reduce_dims = [i for i in range(ndA) if not used[i]]
    if not reduce_dims:
        reduce_dims = [ndA - 1]
    out_shape = [inp[i] for i in range(ndA) if i not in set(reduce_dims)]
    if not out_shape:
        out_shape = [1]
    if not is_static(out_shape):
        return None
    reduce_size = 1
    for d in reduce_dims:
        reduce_size *= inp[d]
    t_in  = mt(inp); t_out = mt(out_shape); fn = _fname(stem)

    # Batch dims (non-reduced) and reduce dims
    batch_dims = [i for i in range(ndA) if i not in set(reduce_dims)]
    nd_batch   = len(batch_dims); nd_red = len(reduce_dims)

    body = _cst_indices(ndA) + _load_dims("input", inp)
    body += ["    %out = memref.alloc() : {}".format(t_out)]
    nsz_f = "{}.0".format(reduce_size)
    body += ["    %nsz = arith.constant {} : f32".format(nsz_f)]

    batch_lvars = ["rm_b{}".format(i) for i in range(nd_batch)]
    for i in range(nd_batch):
        body.append(_open_for(i, batch_lvars[i], "%input_d{}".format(batch_dims[i])))
    oi = _ind(nd_batch)
    batch_idx = ", ".join("%{}".format(v) for v in batch_lvars) if batch_lvars else ""

    # Inner reduction over reduce_dims (single dim supported; multi-dim flattened)
    if nd_red == 1:
        ax = reduce_dims[0]
        in_idx_for_load = lambda rvar: (
            "[" + ", ".join(
                ("%{}".format(batch_lvars[batch_dims.index(d)]) if d in batch_dims
                 else ("%{}".format(rvar))) for d in range(ndA)
            ) + "]"
        )
        body += [
            "{}%zero_f = arith.constant 0.0 : f32".format(oi),
            "{}%rm_sum = scf.for %rm_k = %c0 to %input_d{ax} step %c1"
            " iter_args(%rm_s = %zero_f) -> (f32) {{".format(oi, ax=ax),
            "{}  %rv   = memref.load %input{} : {}".format(
                oi, in_idx_for_load("rm_k"), t_in),
            "{}  %ns   = arith.addf %rm_s, %rv : f32".format(oi),
            "{}  scf.yield %ns : f32".format(oi),
            "{}}}".format(oi),
            "{}%rm_mean = arith.divf %rm_sum, %nsz : f32".format(oi),
            "{}memref.store %rm_mean, %out{} : {}".format(
                oi, "[" + batch_idx + "]" if batch_idx else "[%c0]", t_out),
        ]
    else:
        # Multi-reduce: iterate over all reduce dims with nested loops
        # (simplified: just compute sum over all reduce combos)
        red_lvars = ["rm_r{}".format(i) for i in range(nd_red)]
        body += ["{}%zero_f = arith.constant 0.0 : f32".format(oi)]
        acc_str = "%zero_f"
        # Build nested reduction
        for ri, (rd, rv) in enumerate(zip(reduce_dims, red_lvars)):
            if ri == 0:
                body += ["{}%rm_sum = scf.for %{} = %c0 to %input_d{} step %c1"
                         " iter_args(%rm_s0 = {}) -> (f32) {{".format(
                    oi, rv, rd, acc_str)]
            else:
                outer_ii = oi + "  " * ri
                body += ["{}  %rm_sum{} = scf.for %{} = %c0 to %input_d{} step %c1"
                         " iter_args(%rm_s{} = %rm_s{}) -> (f32) {{".format(
                    oi, ri, rv, rd, ri, ri - 1)]
        # innermost body
        all_lvars = {batch_dims[i]: "%{}".format(batch_lvars[i]) for i in range(nd_batch)}
        for i, rd in enumerate(reduce_dims):
            all_lvars[rd] = "%{}".format(red_lvars[i])
        in_ids = "[" + ", ".join(all_lvars[d] for d in range(ndA)) + "]"
        body += [
            "{}  %rv   = memref.load %input{} : {}".format(oi, in_ids, t_in),
            "{}  %ns   = arith.addf %rm_s{}, %rv : f32".format(oi, nd_red - 1),
            "{}  scf.yield %ns : f32".format(oi),
        ]
        for ri in range(nd_red - 1, -1, -1):
            body += ["{}{}}}".format(oi, "  " * ri)]
            if ri > 0:
                body += ["{}  scf.yield %rm_sum{} : f32".format(oi, ri)]
        body += [
            "{}%rm_mean = arith.divf %rm_sum, %nsz : f32".format(oi),
            "{}memref.store %rm_mean, %out{} : {}".format(
                oi, "[" + batch_idx + "]" if batch_idx else "[%c0]", t_out),
        ]

    for i in range(nd_batch - 1, -1, -1):
        body.append(_close_for(i))
    body += ["    return %out : {}".format(t_out)]
    return _build_module(fn, [("input", inp, "f32")], body, out_shape)

# ---- embedding ----

def gen_embedding(stem, shapes):
    if len(shapes) < 3:
        return None
    idx_dims, table_dims, out_dims = shapes[0], shapes[1], shapes[2]
    if not is_static(idx_dims) or not is_static(table_dims) or not is_static(out_dims):
        return None
    nd_idx = len(idx_dims)
    embed_dim = table_dims[1] if len(table_dims) > 1 else table_dims[0]
    t_idx   = mt(idx_dims, "i64"); t_tbl = mt(table_dims); t_out = mt(out_dims)
    fn = _fname(stem)

    body = _cst_indices(nd_idx + 1) + _load_dims("indices", idx_dims, "i64") + _load_dims("table", table_dims)
    body += ["    %out = memref.alloc() : {}".format(t_out)]

    lvars_idx = ["em_b{}".format(i) for i in range(nd_idx)]
    for i in range(nd_idx):
        body.append(_open_for(i, lvars_idx[i], "%indices_d{}".format(i)))
    batch_ii = _ind(nd_idx)
    body.append(_open_for(nd_idx, "em_d0", "%table_d{}".format(1 if len(table_dims) > 1 else 0)))
    ii  = _ind(nd_idx + 1)
    idx = _idx_str(lvars_idx)
    body += [
        "{}%raw_idx = memref.load %indices{} : {}".format(ii, idx, t_idx),
        "{}%row_idx = arith.index_cast %raw_idx : i64 to index".format(ii),
        "{}%tv      = memref.load %table[%row_idx, %em_d0] : {}".format(ii, t_tbl),
        "{}memref.store %tv, %out[{}, %em_d0] : {}".format(
            ii, ", ".join("%{}".format(v) for v in lvars_idx), t_out),
    ]
    body.append(_close_for(nd_idx))
    for i in range(nd_idx - 1, -1, -1):
        body.append(_close_for(i))
    body += ["    return %out : {}".format(t_out)]
    return _build_module(fn, [("table", table_dims, "f32"), ("indices", idx_dims, "i64")], body, out_dims)

# ---------------------------------------------------------------------------
# Dispatch table + driver
# ---------------------------------------------------------------------------

GENERATORS = {
    "relu":                gen_relu,
    "sigmoid":             gen_sigmoid,
    "gelu":                gen_gelu,
    "elemwise_add":        gen_elemwise_add,
    "softmax":             gen_softmax,
    "matmul":              gen_matmul,
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


def main():
    generated = 0; skipped = 0; skip_reasons = []
    for category, gen_fn in GENERATORS.items():
        src_cat = STABLEHLO_DIR / category
        if not src_cat.exists():
            continue
        out_cat = OUT_DIR / category
        out_cat.mkdir(parents=True, exist_ok=True)
        for src_file in sorted(src_cat.iterdir()):
            if src_file.suffix != ".mlir":
                continue
            stem = src_file.stem
            shapes = shapes_for_category(stem, category)
            if not shapes or not all(is_static(s) for s in shapes if s):
                skipped += 1
                skip_reasons.append((category, stem, "dynamic/empty shapes"))
                continue
            mlir = gen_fn(stem, shapes)
            if mlir is None:
                skipped += 1
                skip_reasons.append((category, stem, "generator returned None"))
                continue
            (out_cat / (stem + ".mlir")).write_text(mlir)
            generated += 1

    total = generated + skipped
    print("Generated {}/{} memref MLIR cases.".format(generated, total))
    if skipped:
        print("  {} skipped:".format(skipped))
        for cat, stem, reason in skip_reasons:
            print("    {}/{}: {}".format(cat, stem, reason))


if __name__ == "__main__":
    main()
