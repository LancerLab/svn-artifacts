#!/usr/bin/env python3
"""Generate StableHLO MLIR benchmarks for the IREE real-world pipeline.

Each case uses StableHLO dialect ops — the natural input format for IREE
when coming from JAX / TensorFlow / XLA-based frameworks.  Only static
shapes are emitted; dynamic-dim cases (those with '?' in the stem) are
skipped because stablehlo broadcast/reshape ops need concrete output
shapes at compile time (mirroring how JAX traces concrete computations).

Validation: iree-compile --iree-input-type=stablehlo --compile-to=input -o -
Inspection: iree-compile --iree-input-type=stablehlo --compile-to=flow  -o -

Output: benchmark/stablehlo/{category}/{stem}.mlir
"""

from pathlib import Path

ROOT       = Path(__file__).parent.parent
CHOREO_DIR = ROOT / "benchmark/choreo"
OUT_DIR    = ROOT / "benchmark/stablehlo"

# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------

def _to_dim(s):
    try:
        return int(s)
    except ValueError:
        return s   # keep raw symbol ("N", "H", "S", ...); _mlir_dim() maps to "?" in types

def _mlir_dim(d):
    return str(d) if isinstance(d, int) else "?"

def parse_shape(s):
    return [_to_dim(p) for p in s.split("x") if p]

def tensor_type(dims, dtype="f32"):
    return "tensor<" + "x".join(_mlir_dim(d) for d in dims) + "x" + dtype + ">"

def is_static(dims):
    return all(isinstance(d, int) for d in dims)

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
        return [s for s in shapes if len(s) >= 2]
    return [parse_shape(r) for r in raw]

def _find_concat_axis(inputs, C):
    nd = len(C)
    # First: try static sum matching
    for d in range(nd):
        c_d = C[d]; in_d = [inp[d] for inp in inputs]
        if isinstance(c_d, int) and all(isinstance(x, int) for x in in_d):
            if sum(in_d) == c_d: return d
    # Fallback: find axis where not all (input + output) dim tokens are identical
    for d in range(nd):
        vals = set(str(inp[d]) for inp in inputs) | {str(C[d])}
        if len(vals) > 1:
            return d
    return nd - 1

def _find_expand_reassoc(A, C):
    ndA, ndC = len(A), len(C)
    def helper(ai, ci):
        if ai == ndA: return [] if ci == ndC else None
        max_size = ndC - ci - (ndA - ai - 1)
        for size in range(1, max_size + 1):
            grp = list(range(ci, ci + size))
            a_dim = A[ai]
            if isinstance(a_dim, int):
                prod = 1; has_dyn = False
                for g in grp:
                    d = C[g]
                    if not isinstance(d, int): has_dyn = True; break
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
    lines = ["module @{} {{".format(fname), header] + body_lines + ["  }", "}"]
    return "\n".join(lines) + "\n"

def _splat_var(val):
    """Return an SSA name that is always a valid (letter-starting) identifier."""
    s = str(val).replace('.', 'p').replace('-', 'm').replace('+', 'pl')
    return "%c_{}".format(s)  # always starts with letter

def _bcast(src_var, src_dims, out_dims, broadcast_dims, out_var):
    """stablehlo.broadcast_in_dim from lower-rank to out_dims."""
    src_t = tensor_type(src_dims)
    out_t = tensor_type(out_dims)
    return "    {} = stablehlo.broadcast_in_dim {}, dims = [{}] : ({}) -> {}".format(
        out_var, src_var, ", ".join(str(d) for d in broadcast_dims), src_t, out_t)


def _find_bcast_dims(src_dims, out_dims):
    """Find broadcast_dims mapping each src axis to the matching out axis by value
    (first unused match). Falls back to trailing-dims alignment."""
    used = [False] * len(out_dims)
    result = []
    for sd in src_dims:
        matched = False
        for i, od in enumerate(out_dims):
            if not used[i] and od == sd:
                result.append(i); used[i] = True; matched = True; break
        if not matched:
            # fallback: just append sequentially
            for i in range(len(out_dims)):
                if not used[i]:
                    result.append(i); used[i] = True; break
    return result

# ---------------------------------------------------------------------------
# Category generators
# ---------------------------------------------------------------------------

# ---- relu ----

def gen_relu(stem, shapes):
    if not shapes: return None
    A = shapes[0]; tA = tensor_type(A)
    fname = stem
    # Use clamp with scalar f32 bounds -- works for static and dynamic shapes
    body = [
        "    %zero   = stablehlo.constant dense<0.0> : tensor<f32>",
        "    %inf    = stablehlo.constant dense<3.4028235e+38> : tensor<f32>",
        "    %result = stablehlo.clamp %zero, %input, %inf : (tensor<f32>, {0}, tensor<f32>) -> {0}".format(tA),
        "    return %result : {}".format(tA),
    ]
    return build_mlir(fname, [("input", A)], body, A)

# ---- sigmoid ----

def gen_sigmoid(stem, shapes):
    if not shapes: return None
    A = shapes[0]; tA = tensor_type(A)
    fname = stem
    body = [
        "    %result = stablehlo.logistic %input : {}".format(tA),
        "    return %result : {}".format(tA),
    ]
    return build_mlir(fname, [("input", A)], body, A)

# ---- gelu (tanh approximation) ----

def gen_gelu(stem, shapes):
    if not shapes: return None
    A = shapes[0]; tA = tensor_type(A)
    if not is_static(A): return None  # constants need static shapes in IREE stablehlo
    fname = stem
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    body = [
        "    %half    = stablehlo.constant dense<5.000000e-01> : {}".format(tA),
        "    %one     = stablehlo.constant dense<1.000000e+00> : {}".format(tA),
        "    %sqrt2pi = stablehlo.constant dense<7.978846e-01> : {}".format(tA),
        "    %coeff   = stablehlo.constant dense<4.471500e-02> : {}".format(tA),
        "    %x2      = stablehlo.multiply %input, %input : {}".format(tA),
        "    %x3      = stablehlo.multiply %x2, %input : {}".format(tA),
        "    %cx3     = stablehlo.multiply %coeff, %x3 : {}".format(tA),
        "    %inner   = stablehlo.add %input, %cx3 : {}".format(tA),
        "    %targ    = stablehlo.multiply %sqrt2pi, %inner : {}".format(tA),
        "    %tv      = stablehlo.tanh %targ : {}".format(tA),
        "    %one_tv  = stablehlo.add %one, %tv : {}".format(tA),
        "    %hx      = stablehlo.multiply %half, %input : {}".format(tA),
        "    %result  = stablehlo.multiply %hx, %one_tv : {}".format(tA),
        "    return %result : {}".format(tA),
    ]
    return build_mlir(fname, [("input", A)], body, A)

# ---- elemwise_add ----

def gen_elemwise_add(stem, shapes):
    if len(shapes) < 2: return None
    A, B_declared = shapes[0], shapes[1]
    C = shapes[2] if len(shapes) >= 3 else A
    if not A or not C: return None
    fname = stem
    tA = tensor_type(A); tC = tensor_type(C)

    ndA, ndC = len(A), len(C)
    # Determine broadcast pattern
    body = []

    if B_declared == A:
        # No broadcast needed -- works for dynamic shapes too
        body.append("    %result = stablehlo.add %input0, %input1 : {}".format(tA))
        body.append("    return %result : {}".format(tA))
        return build_mlir(fname, [("input0", A), ("input1", A)], body, C)

    # B needs broadcasting -- requires broadcast_in_dim which needs static output
    if not is_static(A):
        return None  # broadcast_in_dim to dynamic output is illegal in IREE stablehlo

    # B may have lower rank or same rank with 1-dims
    ndB = len(B_declared)
    bcast_dims = _find_bcast_dims(B_declared, A)
    body.append(_bcast("%input1", B_declared, A, bcast_dims, "%b_bcast"))
    body.append("    %result = stablehlo.add %input0, %b_bcast : {}".format(tA))
    body.append("    return %result : {}".format(tA))
    return build_mlir(fname, [("input0", A), ("input1", B_declared)], body, C)

# ---- softmax ----

def gen_softmax(stem, shapes):
    if not shapes: return None
    A = shapes[0]; ndA = len(A); tA = tensor_type(A)
    if not is_static(A): return None  # broadcast_in_dim requires static output dims
    # Reduce over the last axis
    reduce_dim = ndA - 1
    outer_dims = A[:reduce_dim]
    tOuter = tensor_type(outer_dims) if outer_dims else "tensor<f32>"
    fname = stem

    body = []
    # stablehlo.reduce to get max
    body.append("    %neg_inf = stablehlo.constant dense<-3.402820e+38> : tensor<f32>")
    body.append("    %max_red = stablehlo.reduce(%input init: %neg_inf) across dimensions = [{}] : ({}, tensor<f32>) -> {}".format(
        reduce_dim, tA, tOuter))
    body.append("      reducer(%lhs: tensor<f32>, %rhs: tensor<f32>)  {")
    body.append("        %m = stablehlo.maximum %lhs, %rhs : tensor<f32>")
    body.append("        stablehlo.return %m : tensor<f32>")
    body.append("      }")
    # broadcast max back to full shape
    bcast_dims_outer = list(range(reduce_dim))
    if outer_dims:
        body.append(_bcast("%max_red", outer_dims, A, bcast_dims_outer, "%max_bcast"))
    else:
        body.append("    %max_bcast = stablehlo.broadcast_in_dim %max_red, dims = [] : (tensor<f32>) -> {}".format(tA))
    body.append("    %shifted   = stablehlo.subtract %input, %max_bcast : {}".format(tA))
    body.append("    %exp_vals  = stablehlo.exponential %shifted : {}".format(tA))
    body.append("    %zero_sum  = stablehlo.constant dense<0.0> : tensor<f32>")
    body.append("    %sum_red   = stablehlo.reduce(%exp_vals init: %zero_sum) across dimensions = [{}] : ({}, tensor<f32>) -> {}".format(
        reduce_dim, tA, tOuter))
    body.append("      reducer(%lhs: tensor<f32>, %rhs: tensor<f32>)  {")
    body.append("        %s = stablehlo.add %lhs, %rhs : tensor<f32>")
    body.append("        stablehlo.return %s : tensor<f32>")
    body.append("      }")
    if outer_dims:
        body.append(_bcast("%sum_red", outer_dims, A, bcast_dims_outer, "%sum_bcast"))
    else:
        body.append("    %sum_bcast = stablehlo.broadcast_in_dim %sum_red, dims = [] : (tensor<f32>) -> {}".format(tA))
    body.append("    %result    = stablehlo.divide %exp_vals, %sum_bcast : {}".format(tA))
    body.append("    return %result : {}".format(tA))
    return build_mlir(fname, [("input", A)], body, A)

# ---- matmul ----

def gen_matmul(stem, shapes):
    if len(shapes) < 3: return None
    A, B, C = shapes[0], shapes[1], shapes[2]
    if len(A) == 0 or len(B) == 0: return None
    fname = stem
    ndA = len(A)

    tA = tensor_type(A); tB = tensor_type(B); tC = tensor_type(C)
    body = []

    if ndA == 2:
        # stablehlo.dot: straightforward
        body.append("    %result = stablehlo.dot %input0, %input1 : ({}, {}) -> {}".format(tA, tB, tC))
    elif ndA == 3 and len(B) == 3:
        # batched: dot_general with batch dim 0, contract [2] x [1]
        body.append("    %result = stablehlo.dot_general %input0, %input1,")
        body.append("        batching_dims = [0] x [0],")
        body.append("        contracting_dims = [2] x [1] : ({}, {}) -> {}".format(tA, tB, tC))
    elif ndA == 3 and len(B) == 2:
        # [B, M, K] x [K, N] -> [B, M, N]: no batch dim
        # IREE creates dynamic_reshape internally when A or C are not fully static
        if not is_static(A) or not is_static(C):
            return None
        body.append("    %result = stablehlo.dot_general %input0, %input1,")
        body.append("        batching_dims = [] x [],")
        body.append("        contracting_dims = [2] x [0] : ({}, {}) -> {}".format(tA, tB, tC))
    elif ndA == 4:
        # [B1, B2, M, K] x [B1, B2, K, N] -> batch over [0,1]
        body.append("    %result = stablehlo.dot_general %input0, %input1,")
        body.append("        batching_dims = [0, 1] x [0, 1],")
        body.append("        contracting_dims = [3] x [2] : ({}, {}) -> {}".format(tA, tB, tC))
    else:
        return None

    body.append("    return %result : {}".format(tC))
    return build_mlir(fname, [("input0", A), ("input1", B)], body, C)

# ---- transpose ----

def gen_transpose(stem, shapes):
    if len(shapes) < 2: return None
    A, C = shapes[0], shapes[1]
    if len(A) == 0 or len(C) == 0: return None
    fname = stem; ndA = len(A)
    # Infer permutation from shapes
    perm = list(range(ndA))
    tA = tensor_type(A); tC = tensor_type(C)
    # Try all permutations to find one matching output shape
    from itertools import permutations
    found = None
    for p in permutations(range(ndA)):
        if [A[i] for i in p] == C:
            found = list(p); break
    if found is None: return None
    perm_str = "dense<[{}]> : tensor<{}xi64>".format(", ".join(str(x) for x in found), ndA)
    body = [
        "    %result = stablehlo.transpose %input, dims = [{}] : ({}) -> {}".format(
            ", ".join(str(x) for x in found), tA, tC),
        "    return %result : {}".format(tC),
    ]
    return build_mlir(fname, [("input", A)], body, C)

# ---- reshape ----

def _find_collapse_groups(A, C):
    """For collapse (ndA > ndC): find how groups of A dims map to each C dim.
    Matches same-symbol 1:1 first, then groups remaining A dims into unmatched C dim.
    Returns list-of-groups or None."""
    ndA, ndC = len(A), len(C)
    if ndA <= ndC: return None
    matched_a, matched_c = {}, {}
    for ai, a_d in enumerate(A):
        for ci, c_d in enumerate(C):
            if ci not in matched_c and ai not in matched_a and a_d == c_d:
                matched_a[ai] = ci; matched_c[ci] = ai; break
    unmatched_a = [ai for ai in range(ndA) if ai not in matched_a]
    unmatched_c = [ci for ci in range(ndC) if ci not in matched_c]
    if len(unmatched_c) == 1:
        groups = [None] * ndC
        for ai, ci in matched_a.items(): groups[ci] = [ai]
        groups[unmatched_c[0]] = sorted(unmatched_a)
        if sorted(ai for g in groups for ai in g) == list(range(ndA)):
            return groups
    return None

def _gen_collapse_shape_lines(inp_var, A, groups, tA, ndC):
    """Generate body lines to build a tensor<ndC xi32> output shape for dynamic_reshape (collapse)."""
    lines = []; shape_elems = []
    for ci, grp in enumerate(groups):
        if len(grp) == 1:
            ai = grp[0]
            v = "%_rs_s{}".format(ci)
            if isinstance(A[ai], int):
                lines.append("    {} = stablehlo.constant dense<{}> : tensor<i32>".format(v, A[ai]))
            else:
                lines.append("    {} = stablehlo.get_dimension_size %{}, dim = {} : ({}) -> tensor<i32>".format(v, inp_var, ai, tA))
            shape_elems.append(v)
        else:
            prev = None
            for j, ai in enumerate(grp):
                dv = "%_rs_d{}_{}".format(ci, j)
                if isinstance(A[ai], int):
                    lines.append("    {} = stablehlo.constant dense<{}> : tensor<i32>".format(dv, A[ai]))
                else:
                    lines.append("    {} = stablehlo.get_dimension_size %{}, dim = {} : ({}) -> tensor<i32>".format(dv, inp_var, ai, tA))
                if prev is not None:
                    pv = "%_rs_p{}_{}".format(ci, j)
                    lines.append("    {} = stablehlo.multiply {}, {} : tensor<i32>".format(pv, prev, dv))
                    prev = pv
                else:
                    prev = dv
            shape_elems.append(prev)
    in_types = ", ".join("tensor<i32>" for _ in shape_elems)
    lines.append("    %_rs_shape = stablehlo.concatenate {}, dim = 0 : ({}) -> tensor<{}xi32>".format(
        ", ".join(shape_elems), in_types, ndC))
    return lines, "%_rs_shape"

def gen_reshape(stem, shapes):
    if len(shapes) < 2: return None
    A, C = shapes[0], shapes[1]
    tA = tensor_type(A); tC = tensor_type(C)
    fname = stem

    if is_static(A) and is_static(C):
        body = [
            "    %result = stablehlo.reshape %input : ({}) -> {}".format(tA, tC),
            "    return %result : {}".format(tC),
        ]
        return build_mlir(fname, [("input", A)], body, C)

    # Dynamic: stablehlo.dynamic_reshape is explicitly illegal in IREE stablehlo
    # Also, concatenating rank-0 tensors fails. Skip all dynamic reshape cases.
    return None

    # Dynamic: use dynamic_reshape with computed output shape
    ndA, ndC = len(A), len(C)
    if ndA > ndC:
        # Collapse: find groups of A dims → each C dim
        groups = _find_collapse_groups(A, C)
        if groups is None: return None
        body = ["    // reshape: dynamic collapse using stablehlo.dynamic_reshape"]
        shape_lines, shape_var = _gen_collapse_shape_lines("input", A, groups, tA, ndC)
        body += shape_lines
        body += [
            "    %result = stablehlo.dynamic_reshape %input, {} : ({}, tensor<{}xi32>) -> {}".format(
                shape_var, tA, ndC, tC),
            "    return %result : {}".format(tC),
        ]
        return build_mlir(fname, [("input", A)], body, C)
    elif ndA < ndC:
        # Expand: A has fewer dims; find groups from static expand reassoc
        reassoc = _find_expand_reassoc(A, C)
        if reassoc is None: return None
        # Build shape from C dims (all should be static for expand to work)
        body = ["    // reshape: dynamic expand using stablehlo.dynamic_reshape"]
        shape_elems = []; shape_lines = []
        for ci, d in enumerate(C):
            v = "%_rs_s{}".format(ci)
            if isinstance(d, int):
                shape_lines.append("    {} = stablehlo.constant dense<{}> : tensor<i32>".format(v, d))
                shape_elems.append(v)
            else:
                # Dynamic expand: try to find corresponding A dim
                # Find which A dim maps to this C dim via reassoc
                found_ai = None
                for ai, grp in enumerate(reassoc):
                    if ci in grp and len(grp) == 1:
                        found_ai = ai; break
                if found_ai is not None:
                    shape_lines.append("    {} = stablehlo.get_dimension_size %input, dim = {} : ({}) -> tensor<i32>".format(
                        v, found_ai, tA))
                    shape_elems.append(v)
                else:
                    return None  # Can't compute shape for this output dim
        in_types = ", ".join("tensor<i32>" for _ in shape_elems)
        shape_lines.append("    %_rs_shape = stablehlo.concatenate {}, dim = 0 : ({}) -> tensor<{}xi32>".format(
            ", ".join(shape_elems), in_types, ndC))
        body += shape_lines
        body += [
            "    %result = stablehlo.dynamic_reshape %input, %_rs_shape : ({}, tensor<{}xi32>) -> {}".format(
                tA, ndC, tC),
            "    return %result : {}".format(tC),
        ]
        return build_mlir(fname, [("input", A)], body, C)
    else:
        # Same rank but different shape (permutation / view)
        body = [
            "    %result = stablehlo.dynamic_reshape %input, %_rs_shape : ({}, tensor<{}xi32>) -> {}".format(
                tA, ndC, tC),
        ]
        # Build shape from C dims
        shape_elems = []; shape_lines = []
        for ci, d in enumerate(C):
            v = "%_rs_s{}".format(ci)
            if isinstance(d, int):
                shape_lines.append("    {} = stablehlo.constant dense<{}> : tensor<i32>".format(v, d))
            else:
                shape_lines.append("    {} = stablehlo.get_dimension_size %input, dim = {} : ({}) -> tensor<i32>".format(
                    v, ci, tA))
            shape_elems.append(v)
        in_types = ", ".join("tensor<i32>" for _ in shape_elems)
        shape_lines.append("    %_rs_shape = stablehlo.concatenate {}, dim = 0 : ({}) -> tensor<{}xi32>".format(
            ", ".join(shape_elems), in_types, ndC))
        body2 = ["    // reshape: same-rank dynamic reshape"] + shape_lines + [
            "    %result = stablehlo.dynamic_reshape %input, %_rs_shape : ({}, tensor<{}xi32>) -> {}".format(
                tA, ndC, tC),
            "    return %result : {}".format(tC),
        ]
        return build_mlir(fname, [("input", A)], body2, C)

# ---- concat ----

def gen_concat(stem, shapes):
    if len(shapes) < 3: return None
    inputs = shapes[:-1]; C = shapes[-1]
    if not inputs or not C: return None
    axis = _find_concat_axis(inputs, C)
    fname = stem; tC = tensor_type(C)
    arg_list = [("in{}".format(ii), inp) for ii, inp in enumerate(inputs)]
    in_vars = ", ".join("%in{}".format(ii) for ii in range(len(inputs)))
    in_types = ", ".join(tensor_type(inp) for inp in inputs)
    body = [
        "    %result = stablehlo.concatenate {}, dim = {} : ({}) -> {}".format(
            in_vars, axis, in_types, tC),
        "    return %result : {}".format(tC),
    ]
    return build_mlir(fname, arg_list, body, C)

# ---- batch_norm ----

def gen_batch_norm(stem, shapes):
    if len(shapes) < 3: return None
    inp = shapes[0]; gamma_dims = shapes[1]; beta_dims = shapes[2] if len(shapes) >= 4 else shapes[1]
    C_out = shapes[-1]
    if not is_static(gamma_dims): return None  # need static gamma for mean/var constants
    nd = len(inp)
    # Find channel axis
    ch_axis = 1
    for ai, av in enumerate(inp):
        if av == gamma_dims[0]: ch_axis = ai; break
    tA = tensor_type(inp); tG = tensor_type(gamma_dims); tBt = tensor_type(beta_dims)
    fname = stem
    # stablehlo.batch_norm_inference has no custom assembly form; use generic form.
    body = [
        "    %mean    = stablehlo.constant dense<0.0> : {}".format(tG),
        "    %var     = stablehlo.constant dense<1.0> : {}".format(tG),
        "    %result  = \"stablehlo.batch_norm_inference\"(%input, %gamma, %beta, %mean, %var)",
        "        {{epsilon = 1.000000e-05 : f32, feature_index = {} : i64}}".format(ch_axis),
        "        : ({}, {}, {}, {}, {}) -> {}".format(tA, tG, tBt, tG, tG, tA),
        "    return %result : {}".format(tA),
    ]
    return build_mlir(fname, [("input", inp), ("gamma", gamma_dims), ("beta", beta_dims)], body, C_out)

# ---- layer_normalization ----

def gen_layer_norm(stem, shapes):
    if len(shapes) < 2: return None
    inp = shapes[0]; gamma_dims = shapes[1]
    beta_dims = shapes[2] if len(shapes) >= 3 else gamma_dims
    if not is_static(gamma_dims): return None  # need static gamma/beta dims
    nd = len(inp)
    norm_rank = len(gamma_dims); batch_rank = nd - norm_rank
    tA  = tensor_type(inp); tG = tensor_type(gamma_dims); tBt = tensor_type(beta_dims)
    outer_shape = inp[:batch_rank]
    tOuter = tensor_type(outer_shape) if outer_shape else "tensor<f32>"
    reduce_dims = list(range(batch_rank, nd))
    reduce_size = 1
    for d in inp[batch_rank:]: reduce_size *= d
    bcast_dims_outer = list(range(batch_rank))
    bcast_dims_norm  = list(range(batch_rank, nd))
    fname = stem

    body = []
    # 1. Reduce sum and sum-of-squares over norm dims
    body.append("    %zero_f   = stablehlo.constant dense<0.0> : tensor<f32>")
    body.append("    %sum_red  = stablehlo.reduce(%input init: %zero_f) across dimensions = [{}] : ({}, tensor<f32>) -> {}".format(
        ", ".join(str(d) for d in reduce_dims), tA, tOuter))
    body.append("      reducer(%lhs: tensor<f32>, %rhs: tensor<f32>)  {")
    body.append("        %s = stablehlo.add %lhs, %rhs : tensor<f32>")
    body.append("        stablehlo.return %s : tensor<f32>")
    body.append("      }")
    body.append("    %sq_in    = stablehlo.multiply %input, %input : {}".format(tA))
    body.append("    %sq_red   = stablehlo.reduce(%sq_in init: %zero_f) across dimensions = [{}] : ({}, tensor<f32>) -> {}".format(
        ", ".join(str(d) for d in reduce_dims), tA, tOuter))
    body.append("      reducer(%lhs: tensor<f32>, %rhs: tensor<f32>)  {")
    body.append("        %s2 = stablehlo.add %lhs, %rhs : tensor<f32>")
    body.append("        stablehlo.return %s2 : tensor<f32>")
    body.append("      }")
    # 2. Compute mean and inv_std
    # broadcast_in_dim requires static output; skip if outer_shape has dynamic dims
    if not is_static(outer_shape):
        return None  # broadcast would target dynamic type -- illegal in IREE stablehlo
    if not isinstance(reduce_size, int):
        return None  # dynamic norm_size not supported
    body.append("    %nsz      = stablehlo.constant dense<{}.0> : {}".format(reduce_size, tOuter))
    body.append("    %eps      = stablehlo.constant dense<1.0e-05> : {}".format(tOuter))
    body.append("    %mean     = stablehlo.divide %sum_red, %nsz : {}".format(tOuter))
    body.append("    %mean_sq  = stablehlo.divide %sq_red, %nsz : {}".format(tOuter))
    body.append("    %m2       = stablehlo.multiply %mean, %mean : {}".format(tOuter))
    body.append("    %variance = stablehlo.subtract %mean_sq, %m2 : {}".format(tOuter))
    body.append("    %var_eps  = stablehlo.add %variance, %eps : {}".format(tOuter))
    body.append("    %inv_std  = stablehlo.rsqrt %var_eps : {}".format(tOuter))
    # 3. Broadcast mean/inv_std to full shape and normalize
    if outer_shape:
        body.append(_bcast("%mean",    outer_shape, inp, bcast_dims_outer, "%mean_b"))
        body.append(_bcast("%inv_std", outer_shape, inp, bcast_dims_outer, "%istd_b"))
    else:
        body.append("    %mean_b = stablehlo.broadcast_in_dim %mean, dims = [] : (tensor<f32>) -> {}".format(tA))
        body.append("    %istd_b = stablehlo.broadcast_in_dim %inv_std, dims = [] : (tensor<f32>) -> {}".format(tA))
    body.append("    %cent     = stablehlo.subtract %input, %mean_b : {}".format(tA))
    body.append("    %normed   = stablehlo.multiply %cent, %istd_b : {}".format(tA))
    # 4. Scale and shift
    body.append(_bcast("%gamma", gamma_dims, inp, bcast_dims_norm, "%g_b"))
    body.append(_bcast("%beta",  beta_dims,  inp, bcast_dims_norm, "%b_b"))
    body.append("    %scaled   = stablehlo.multiply %normed, %g_b : {}".format(tA))
    body.append("    %result   = stablehlo.add %scaled, %b_b : {}".format(tA))
    body.append("    return %result : {}".format(tA))
    return build_mlir(fname, [("input", inp), ("gamma", gamma_dims), ("beta", beta_dims)], body, inp)

# ---- conv2d ----

def gen_conv2d(stem, shapes):
    import math as _math
    if len(shapes) < 3: return None
    inp, filt, out = shapes[0], shapes[1], shapes[2]
    if not is_static(filt): return None  # filter must be static for window attrs
    if len(inp) != 4 or len(filt) != 4: return None
    # stablehlo.convolution with dynamic spatial dims is explicitly illegal in IREE
    if not isinstance(inp[2], int) or not isinstance(inp[3], int): return None
    kH = filt[2]; kW = filt[3]
    # Infer stride: use spatial dims when static, else default to 1 (same-size) or 2 (halving)
    if isinstance(inp[2], int) and isinstance(out[2], int) and out[2] > 0:
        strides = max(1, inp[2] // out[2])
    elif inp[2] == out[2]:   # same symbolic dim → stride=1
        strides = 1
    else:
        strides = 1  # conservative default for _S_P_D dynamic cases
    # Compute padding
    def _req_pad(in_sz, out_sz, k, s):
        if not isinstance(in_sz, int) or not isinstance(out_sz, int):
            return 1 if k > 1 else 0  # default same-size padding for 3×3, 0 for 1×1
        req = (out_sz - 1) * s + k
        needed = req - in_sz
        return max(0, _math.ceil(needed / 2))
    pad_h = _req_pad(inp[2], out[2], kH, strides)
    pad_w = _req_pad(inp[3], out[3], kW, strides)
    tA = tensor_type(inp); tF = tensor_type(filt); tC = tensor_type(out)
    fname = stem
    # Use the generic form to avoid custom assembly parser quirks.
    strd_dense = "dense<[{s}, {s}]> : tensor<2xi64>".format(s=strides)
    lhs_d = "dense<[1, 1]> : tensor<2xi64>"
    pad_dense = "dense<[[{ph}, {ph}], [{pw}, {pw}]]> : tensor<2x2xi64>".format(ph=pad_h, pw=pad_w)
    body = []
    body.append("    %result = \"stablehlo.convolution\"(%input, %filter) {")
    body.append("        window_strides = {},".format(strd_dense))
    body.append("        padding = {},".format(pad_dense))
    body.append("        lhs_dilation = {},".format(lhs_d))
    body.append("        rhs_dilation = {},".format(lhs_d))
    body.append("        dimension_numbers = #stablehlo.conv<[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]>,")
    body.append("        feature_group_count = 1 : i64,")
    body.append("        batch_group_count = 1 : i64}")
    body.append("        : ({}, {}) -> {}".format(tA, tF, tC))
    body.append("    return %result : {}".format(tC))
    return build_mlir(fname, [("input", inp), ("filter", filt)], body, out)

# ---- max_pool2d ----

def gen_max_pool2d(stem, shapes):
    import re as _re
    if len(shapes) < 2: return None
    inp, out_shape = shapes[0], shapes[1]
    if len(inp) < 4 or len(out_shape) < 4: return None
    # stablehlo.reduce_window is explicitly illegal in IREE for any dynamic input
    if not is_static(inp): return None
    # Infer stride from static spatial dims; fallback: extract d{N} from stem
    if isinstance(inp[2], int) and isinstance(out_shape[2], int) and out_shape[2] > 0:
        strides = max(1, inp[2] // out_shape[2])
    else:
        # Try to extract denominator from stem token like "Hd2", "Hd5", "Sd3"
        denom = None
        for part in stem.split("_"):
            if "x" not in part: continue
            for tok in part.split("x"):
                m = _re.search(r'd(\d+)$', tok)
                if m: denom = int(m.group(1)); break
            if denom: break
        strides = denom if denom else 2
    kH = kW = strides
    tA = tensor_type(inp); tC = tensor_type(out_shape)
    fname = stem
    # reduce_window with max combiner, NCHW: window over spatial dims only
    nd = len(inp)
    body = []
    body.append("    %neg_inf = stablehlo.constant dense<-3.402820e+38> : tensor<f32>")
    # stablehlo.reduce_window has no custom assembly form; use generic.
    window_dims = [1] * (nd - 2) + [kH, kW]
    window_strd = [1] * (nd - 2) + [strides, strides]
    dims_dense  = "dense<[{}]> : tensor<{}xi64>".format(", ".join(str(x) for x in window_dims), nd)
    strd_dense  = "dense<[{}]> : tensor<{}xi64>".format(", ".join(str(x) for x in window_strd), nd)
    pad_dense   = "dense<0> : tensor<{}x2xi64>".format(nd)
    dil_dense   = "dense<1> : tensor<{}xi64>".format(nd)
    body.append("    %result = \"stablehlo.reduce_window\"(%input, %neg_inf) ({")
    body.append("      ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):")
    body.append("        %m = stablehlo.maximum %lhs, %rhs : tensor<f32>")
    body.append("        \"stablehlo.return\"(%m) : (tensor<f32>) -> ()")
    body.append("    }}) {{window_dimensions = {}, window_strides = {},".format(dims_dense, strd_dense))
    body.append("       padding = {}, base_dilations = {}, window_dilations = {}}}".format(
        pad_dense, dil_dense, dil_dense))
    body.append("        : ({}, tensor<f32>) -> {}".format(tA, tC))
    body.append("    return %result : {}".format(tC))
    return build_mlir(fname, [("input", inp)], body, out_shape)

# ---- reduce_mean ----

def gen_reduce_mean(stem, shapes):
    if len(shapes) < 2: return None
    inp, out_declared = shapes[0], shapes[-1]
    ndA = len(inp)
    # Find reduce dims: match all out_declared dims (including symbolic) against inp
    used = [False] * ndA
    for od in out_declared:
        for i in range(ndA):
            if not used[i] and inp[i] == od:
                used[i] = True; break
    reduce_dims = [i for i in range(ndA) if not used[i]]
    if not reduce_dims: reduce_dims = [ndA - 1]
    out_shape = [inp[i] for i in range(ndA) if i not in set(reduce_dims)]
    if not out_shape: out_shape = [1]
    tA = tensor_type(inp); tOut = tensor_type(out_shape)
    fname = stem
    body = []
    body.append("    %zero     = stablehlo.constant dense<0.0> : tensor<f32>")
    body.append("    %sum_red  = stablehlo.reduce(%input init: %zero) across dimensions = [{}] : ({}, tensor<f32>) -> {}".format(
        ", ".join(str(d) for d in reduce_dims), tA, tOut))
    body.append("      reducer(%lhs: tensor<f32>, %rhs: tensor<f32>)  {")
    body.append("        %s = stablehlo.add %lhs, %rhs : tensor<f32>")
    body.append("        stablehlo.return %s : tensor<f32>")
    body.append("      }")
    # broadcast_in_dim requires static output -- skip dynamic output cases
    if not is_static(out_shape):
        return None  # division would broadcast scalar to dynamic tensor
    reduce_size = 1
    for d in reduce_dims:
        if not isinstance(inp[d], int):
            return None  # dynamic reduce dim: can't compute scalar divisor at compile time
        reduce_size *= inp[d]
    body.append("    %nsz      = stablehlo.constant dense<{}.0> : {}".format(reduce_size, tOut))
    body.append("    %result   = stablehlo.divide %sum_red, %nsz : {}".format(tOut))
    body.append("    return %result : {}".format(tOut))
    return build_mlir(fname, [("input", inp)], body, out_shape)

# ---- embedding ----

def gen_embedding(stem, shapes):
    if len(shapes) < 3: return None
    idx_dims, table_dims, out_dims = shapes[0], shapes[1], shapes[2]
    # embed_dim (table second dim) must be a static integer
    embed_dim = table_dims[1] if len(table_dims) > 1 else table_dims[0]
    if not isinstance(embed_dim, int): return None
    tIdx = tensor_type(idx_dims, "i32")
    tTable = tensor_type(table_dims); tOut = tensor_type(out_dims)
    fname = stem
    # stablehlo.gather: indices are idx_dims, table is table_dims, output per index is embed_dim
    nd_idx = len(idx_dims)
    # Validate: output shape (excluding embed_dim) must match idx_dims
    expected_prefix = list(idx_dims)
    actual_prefix = list(out_dims[:-1])
    if len(expected_prefix) != len(actual_prefix): return None
    for ep, ap in zip(expected_prefix, actual_prefix):
        if ep != ap: return None  # inconsistent batch/seq dims
    body = []
    # stablehlo.gather has no custom assembly form; use generic.
    offset_str = ", ".join(str(d) for d in range(nd_idx, nd_idx + 1))
    body.append("    %result = \"stablehlo.gather\"(%table, %indices)")
    body.append("        {dimension_numbers = #stablehlo.gather<")
    body.append("            offset_dims = [{}],".format(offset_str))
    body.append("            collapsed_slice_dims = [0],")
    body.append("            start_index_map = [0],")
    body.append("            index_vector_dim = {}>,".format(nd_idx))
    body.append("         slice_sizes = dense<[1, {}]> : tensor<2xi64>,".format(embed_dim))
    body.append("         indices_are_sorted = false}")
    body.append("        : ({}, {}) -> {}".format(tTable, tIdx, tOut))
    body.append("    return %result : {}".format(tOut))
    return build_mlir(fname, [("table", table_dims), ("indices", idx_dims, "i32")], body, out_dims)

# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

GENERATORS = {
    "relu":               gen_relu,
    "sigmoid":            gen_sigmoid,
    "gelu":               gen_gelu,
    "elemwise_add":       gen_elemwise_add,
    "softmax":            gen_softmax,
    "matmul":             gen_matmul,
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

def main():
    generated = 0; skipped = 0; skip_reasons = []
    for category, gen_fn in GENERATORS.items():
        choreo_cat = CHOREO_DIR / category
        if not choreo_cat.exists():
            continue
        out_cat = OUT_DIR / category
        out_cat.mkdir(parents=True, exist_ok=True)
        for case_file in sorted(choreo_cat.iterdir()):
            if case_file.suffix not in (".co",) and not case_file.is_dir():
                continue
            stem = case_file.stem if case_file.suffix else case_file.name
            shapes = shapes_for_category(stem, category)
            if not shapes:
                skipped += 1
                skip_reasons.append((category, stem, "no shapes"))
                continue
            mlir = gen_fn(stem, shapes)
            if mlir is None:
                skipped += 1
                skip_reasons.append((category, stem, "generator returned None"))
                continue
            out_path = out_cat / (stem + ".mlir")
            out_path.write_text(mlir)
            generated += 1

    total = generated + skipped
    print("Generated {}/{} StableHLO MLIR cases.".format(generated, total))
    print("  {} cases skipped (dynamic dims or unsupported).".format(skipped))
    if skipped > 0 and len(skip_reasons) <= 20:
        for cat, stem, reason in skip_reasons:
            print("    skip {}/{}: {}".format(cat, stem, reason))

if __name__ == "__main__":
    main()
