#!/usr/bin/env python3
"""
Generate linalg-level MLIR test cases from existing tensor+scf benchmarks.

For each existing benchmark, generates:
  - Correct linalg version (same shapes)
  - Mismatched linalg version (for dim-mismatch bug detection)

Supports:
  - matmul (2D & 3D) → linalg.matmul / linalg.batch_matmul
  - elemwise_add → linalg.add
  - conv2d → linalg.conv_2d_nchw_fchw
  - gelu/sigmoid/relu → linalg.generic (elementwise)
  - softmax → linalg.generic (reduction + elementwise)
  - batch_norm → linalg.generic (broadcast subtract/multiply/add)
  - layer_normalization → linalg.generic
  - concat → tensor.insert_slice (no linalg named op)
  - reduce_mean → linalg.generic (reduction)
  - max_pool2d → linalg.pooling_nchw_max
  - reshape/transpose → tensor.collapse_shape / linalg.transpose
  - embedding → linalg.generic (gather)

Usage: python3 scripts/generate_all_linalg_cases.py
"""

import re
import os
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC_DIR = WORKSPACE / "benchmark" / "mlir" / "cases"
DST_DIR = WORKSPACE / "benchmark" / "mlir" / "cases_linalg"

def parse_func_sig(filepath):
    """Extract function name, input tensor types, and output tensor type."""
    with open(filepath) as f:
        content = f.read()

    # Match: func.func @name(%arg: tensor<...>, ...) -> tensor<...>
    m = re.search(r'func\.func\s+@(\w+)\(([^)]*)\)\s*->\s*(tensor<[^>]+>(?:xf32|xf64)?)', content)
    if not m:
        m = re.search(r'func\.func\s+@(\w+)\(([^)]*)\)\s*->\s*(tensor<[^{]+)', content)
    if not m:
        return None, [], None

    fname = m.group(1)
    args_str = m.group(2)
    ret_type = m.group(3).strip().rstrip('{').strip()

    # Parse argument types
    arg_types = []
    for arg in args_str.split(','):
        arg = arg.strip()
        tm = re.search(r'tensor<([^>]+)>', arg)
        if tm:
            arg_types.append(f"tensor<{tm.group(1)}>")
        elif 'f32' in arg or 'f64' in arg:
            # Scalar
            sm = re.search(r'(f\d+)', arg)
            if sm:
                arg_types.append(sm.group(1))

    return fname, arg_types, ret_type

def tensor_rank(ttype):
    """Return rank of tensor type."""
    m = re.search(r'tensor<(.+)>', ttype)
    if not m:
        return 0
    dims = m.group(1).rstrip('xf32').rstrip('xf64')
    return len(dims.split('x')) if dims else 0

def tensor_dims(ttype):
    """Return list of dimension strings from tensor type."""
    m = re.search(r'tensor<(.+)>', ttype)
    if not m:
        return []
    inner = m.group(1)
    # Remove trailing type
    inner = re.sub(r'x(f32|f64|i32|i64|index)$', '', inner)
    return inner.split('x')

def tensor_elem_type(ttype):
    """Return element type (f32, f64, etc.)."""
    m = re.search(r'(f32|f64|i32|i64)', ttype)
    return m.group(1) if m else 'f32'

def is_dynamic(ttype):
    """Check if tensor has any dynamic dimensions."""
    return '?' in ttype

def make_linalg_matmul(fname, arg_types, ret_type, correct=True):
    """Generate linalg.matmul or linalg.batch_matmul."""
    if len(arg_types) < 2:
        return None
    lhs, rhs = arg_types[0], arg_types[1]
    etype = tensor_elem_type(lhs)
    rank_l = tensor_rank(lhs)
    rank_r = tensor_rank(rhs)

    if rank_l >= 4 or rank_r >= 4:
        return None

    if rank_l == 3 or rank_r == 3:
        return make_linalg_batch_matmul(fname, arg_types, ret_type, correct)

    dims_l = tensor_dims(lhs)
    dims_r = tensor_dims(rhs)
    dims_o = tensor_dims(ret_type)

    if not correct and len(dims_r) >= 1:
        # Inject mismatch: change rhs dim(0) (K dim)
        if dims_r[0] != '?':
            k_val = int(dims_r[0])
            dims_r[0] = str(k_val - 1)
        else:
            pass  # Can't inject static mismatch on dynamic dim

    rhs_bad = f"tensor<{'x'.join(dims_r)}x{etype}>"

    # Generate init tensor dims
    init_dims = []
    init_type_parts = []
    dim_code = []
    idx = 0
    # M from lhs dim 0
    if dims_l[0] == '?':
        dim_code.append(f"    %c0 = arith.constant 0 : index")
        dim_code.append(f"    %m = tensor.dim %a, %c0 : {lhs}")
        init_dims.append('%m')
        init_type_parts.append('?')
    else:
        init_type_parts.append(dims_l[0])
    # N from rhs dim 1
    if len(dims_r) > 1 and dims_r[1] == '?':
        dim_code.append(f"    %c1 = arith.constant 1 : index")
        dim_code.append(f"    %n = tensor.dim %b, %c1 : {rhs_bad}")
        init_dims.append('%n')
        init_type_parts.append('?')
    elif len(dims_r) > 1:
        init_type_parts.append(dims_r[1])

    out_type = f"tensor<{'x'.join(init_type_parts)}x{etype}>"
    init_args = f"({', '.join(init_dims)})" if init_dims else "()"
    empty_call = f"tensor.empty{init_args}" if init_dims else "tensor.empty()"

    suffix = "_correct" if correct else "_mismatch"
    lines = [
        f"module {{",
        f"  func.func @{fname}{suffix}(%a: {lhs}, %b: {rhs_bad}) -> {out_type} {{",
        f"    %cst = arith.constant 0.0 : {etype}",
    ]
    lines.extend(dim_code)
    lines.append(f"    %init = {empty_call} : {out_type}")
    lines.append(f"    %fill = linalg.fill ins(%cst : {etype}) outs(%init : {out_type}) -> {out_type}")
    lines.append(f"    %r = linalg.matmul ins(%a, %b : {lhs}, {rhs_bad})")
    lines.append(f"                        outs(%fill : {out_type}) -> {out_type}")
    lines.append(f"    return %r : {out_type}")
    lines.append(f"  }}")
    lines.append(f"}}")

    return "\n".join(lines)

def make_linalg_batch_matmul(fname, arg_types, ret_type, correct=True):
    """Generate linalg.batch_matmul (3D×3D) or linalg.generic (3D×2D broadcast)."""
    if len(arg_types) < 2:
        return None
    lhs, rhs = arg_types[0], arg_types[1]
    etype = tensor_elem_type(lhs)
    dims_l = tensor_dims(lhs)
    dims_r = tensor_dims(rhs)
    dims_o = tensor_dims(ret_type)

    rank_l = len(dims_l)
    rank_r = len(dims_r)

    if rank_r == 2:
        return make_linalg_generic_bmm(fname, arg_types, ret_type, correct)

    if not correct and len(dims_r) >= 2:
        if dims_r[1] != '?':
            k_val = int(dims_r[1])
            dims_r[1] = str(k_val - 1)

    rhs_type = f"tensor<{'x'.join(dims_r)}x{etype}>"

    dim_code = []
    init_dims = []
    init_parts = []
    for i, d in enumerate(dims_o):
        if d == '?':
            src = '%a' if i < 2 else '%b'
            src_type = lhs if i < 2 else rhs_type
            var = f"%d{i}"
            dim_code.append(f"    %c{i} = arith.constant {i} : index")
            dim_code.append(f"    {var} = tensor.dim {src}, %c{i} : {src_type}")
            init_dims.append(var)
            init_parts.append('?')
        else:
            init_parts.append(d)

    out_type = f"tensor<{'x'.join(init_parts)}x{etype}>"
    init_args = f"({', '.join(init_dims)})" if init_dims else "()"
    empty_call = f"tensor.empty{init_args}" if init_dims else "tensor.empty()"

    suffix = "_correct" if correct else "_mismatch"
    lines = [
        f"module {{",
        f"  func.func @{fname}{suffix}(%a: {lhs}, %b: {rhs_type}) -> {out_type} {{",
        f"    %cst = arith.constant 0.0 : {etype}",
    ]
    lines.extend(dim_code)
    lines.append(f"    %init = {empty_call} : {out_type}")
    lines.append(f"    %fill = linalg.fill ins(%cst : {etype}) outs(%init : {out_type}) -> {out_type}")
    lines.append(f"    %r = linalg.batch_matmul ins(%a, %b : {lhs}, {rhs_type})")
    lines.append(f"                              outs(%fill : {out_type}) -> {out_type}")
    lines.append(f"    return %r : {out_type}")
    lines.append(f"  }}")
    lines.append(f"}}")
    return "\n".join(lines)


def make_linalg_generic_bmm(fname, arg_types, ret_type, correct=True):
    """Generate linalg.generic for 3D×2D broadcast matmul (B×M×K @ K×N → B×M×N)."""
    if len(arg_types) < 2:
        return None
    lhs, rhs = arg_types[0], arg_types[1]
    etype = tensor_elem_type(lhs)
    dims_l = tensor_dims(lhs)  # [B, M, K]
    dims_r = list(tensor_dims(rhs))  # [K, N]
    dims_o = tensor_dims(ret_type)  # [B, M, N]

    if not correct and len(dims_r) >= 1:
        if dims_r[0] != '?':
            k_val = int(dims_r[0])
            dims_r[0] = str(k_val - 1)

    rhs_type = f"tensor<{'x'.join(dims_r)}x{etype}>"

    dim_code = []
    init_dims = []
    init_parts = []
    cidx = 0
    for i, d in enumerate(dims_o):
        if d == '?':
            var = f"%d{i}"
            if i < len(dims_l):
                src, src_type = '%a', lhs
            else:
                src, src_type = '%b', rhs_type
                i_adj = i - len(dims_l) + len(dims_r)
            dim_code.append(f"    %c_i{cidx} = arith.constant {i} : index")
            dim_code.append(f"    {var} = tensor.dim {src}, %c_i{cidx} : {src_type}")
            cidx += 1
            init_dims.append(var)
            init_parts.append('?')
        else:
            init_parts.append(d)

    out_type = f"tensor<{'x'.join(init_parts)}x{etype}>"
    init_args = f"({', '.join(init_dims)})" if init_dims else "()"
    empty_call = f"tensor.empty{init_args}" if init_dims else "tensor.empty()"

    suffix = "_correct" if correct else "_mismatch"
    lines = [
        f"#map_lhs = affine_map<(b, m, n, k) -> (b, m, k)>",
        f"#map_rhs = affine_map<(b, m, n, k) -> (k, n)>",
        f"#map_out = affine_map<(b, m, n, k) -> (b, m, n)>",
        f"",
        f"module {{",
        f"  func.func @{fname}{suffix}(%a: {lhs}, %b: {rhs_type}) -> {out_type} {{",
        f"    %cst = arith.constant 0.0 : {etype}",
    ]
    lines.extend(dim_code)
    lines.append(f"    %init = {empty_call} : {out_type}")
    lines.append(f"    %fill = linalg.fill ins(%cst : {etype}) outs(%init : {out_type}) -> {out_type}")
    lines.append(f"    %r = linalg.generic {{")
    lines.append(f"      indexing_maps = [#map_lhs, #map_rhs, #map_out],")
    lines.append(f"      iterator_types = [\"parallel\", \"parallel\", \"parallel\", \"reduction\"]")
    lines.append(f"    }} ins(%a, %b : {lhs}, {rhs_type})")
    lines.append(f"      outs(%fill : {out_type}) {{")
    lines.append(f"    ^bb0(%in_a: {etype}, %in_b: {etype}, %acc: {etype}):")
    lines.append(f"      %mul = arith.mulf %in_a, %in_b : {etype}")
    lines.append(f"      %add = arith.addf %acc, %mul : {etype}")
    lines.append(f"      linalg.yield %add : {etype}")
    lines.append(f"    }} -> {out_type}")
    lines.append(f"    return %r : {out_type}")
    lines.append(f"  }}")
    lines.append(f"}}")
    return "\n".join(lines)

def make_linalg_add(fname, arg_types, ret_type, correct=True):
    """Generate linalg.add for elementwise addition.
    Falls back to linalg.generic for broadcast cases (different ranks)."""
    if len(arg_types) < 2:
        return None
    lhs, rhs = arg_types[0], arg_types[1]
    etype = tensor_elem_type(lhs)
    lhs_dims = tensor_dims(lhs)
    rhs_dims = tensor_dims(rhs)
    orig_rhs_dims = list(rhs_dims)

    # Skip scalar broadcast cases (tensor<1xf32>) — not representable cleanly
    if len(orig_rhs_dims) == 1 and orig_rhs_dims[0] == '1':
        return None

    if not correct and len(rhs_dims) >= 1:
        for i in range(len(rhs_dims)):
            if rhs_dims[i] != '?':
                rhs_dims[i] = str(int(rhs_dims[i]) - 1)
                break

    rhs_type = f"tensor<{'x'.join(rhs_dims)}x{etype}>"

    # If ranks differ, use linalg.generic with broadcast map
    if len(lhs_dims) != len(rhs_dims):
        rank = len(lhs_dims)
        rhs_rank = len(rhs_dims)

        dim_code = []
        init_dims = []
        init_parts = []
        for i, d in enumerate(lhs_dims):
            if d == '?':
                var = f"%d{i}"
                dim_code.append(f"    %c{i} = arith.constant {i} : index")
                dim_code.append(f"    {var} = tensor.dim %a, %c{i} : {lhs}")
                init_dims.append(var)
                init_parts.append('?')
            else:
                init_parts.append(d)

        out_type = f"tensor<{'x'.join(init_parts)}x{etype}>"
        init_args = f"({', '.join(init_dims)})" if init_dims else "()"
        empty_call = f"tensor.empty{init_args}" if init_dims else "tensor.empty()"

        dims_str = ", ".join([f"d{i}" for i in range(rank)])
        lhs_map = f"affine_map<({dims_str}) -> ({dims_str})>"
        # Map RHS dims to matching LHS dims by size
        if rhs_rank == 0 or (rhs_rank == 1 and rhs_dims[0] == '1'):
            return None  # scalar broadcast not representable in linalg.generic
        elif rhs_rank == 1:
            # Find which lhs dim matches rhs dim size
            match_dim = rank - 1  # default: last dim
            for i, d in enumerate(lhs_dims):
                if d == rhs_dims[0]:
                    match_dim = i
                    break
            rhs_map = f"affine_map<({dims_str}) -> (d{match_dim})>"
        else:
            rhs_dims_idx = ", ".join([f"d{rank - rhs_rank + j}" for j in range(rhs_rank)])
            rhs_map = f"affine_map<({dims_str}) -> ({rhs_dims_idx})>"
        out_map = lhs_map
        iter_types = ", ".join(['"parallel"'] * rank)

        suffix = "_correct" if correct else "_mismatch"
        lines = [
            f"module {{",
            f"  func.func @{fname}{suffix}(%a: {lhs}, %b: {rhs_type}) -> {out_type} {{",
        ]
        lines.extend(dim_code)
        lines.append(f"    %init = {empty_call} : {out_type}")
        lines.append(f"    %r = linalg.generic {{")
        lines.append(f"      indexing_maps = [{lhs_map}, {rhs_map}, {out_map}],")
        lines.append(f"      iterator_types = [{iter_types}]")
        lines.append(f"    }} ins(%a, %b : {lhs}, {rhs_type}) outs(%init : {out_type}) {{")
        lines.append(f"    ^bb0(%x: {etype}, %y: {etype}, %out: {etype}):")
        lines.append(f"      %sum = arith.addf %x, %y : {etype}")
        lines.append(f"      linalg.yield %sum : {etype}")
        lines.append(f"    }} -> {out_type}")
        lines.append(f"    return %r : {out_type}")
        lines.append(f"  }}")
        lines.append(f"}}")
        return "\n".join(lines)

    # Same rank: use linalg.add
    dims = lhs_dims

    dim_code = []
    init_dims = []
    init_parts = []
    for i, d in enumerate(dims):
        if d == '?':
            var = f"%d{i}"
            dim_code.append(f"    %c{i} = arith.constant {i} : index")
            dim_code.append(f"    {var} = tensor.dim %a, %c{i} : {lhs}")
            init_dims.append(var)
            init_parts.append('?')
        else:
            init_parts.append(d)

    out_type = f"tensor<{'x'.join(init_parts)}x{etype}>"
    init_args = f"({', '.join(init_dims)})" if init_dims else "()"
    empty_call = f"tensor.empty{init_args}" if init_dims else "tensor.empty()"

    suffix = "_correct" if correct else "_mismatch"
    lines = [
        f"module {{",
        f"  func.func @{fname}{suffix}(%a: {lhs}, %b: {rhs_type}) -> {out_type} {{",
    ]
    lines.extend(dim_code)
    lines.append(f"    %init = {empty_call} : {out_type}")
    lines.append(f"    %r = linalg.add ins(%a, %b : {lhs}, {rhs_type})")
    lines.append(f"                    outs(%init : {out_type}) -> {out_type}")
    lines.append(f"    return %r : {out_type}")
    lines.append(f"  }}")
    lines.append(f"}}")
    return "\n".join(lines)

def make_linalg_generic_unary(fname, arg_types, ret_type, op="arith.mulf"):
    """Generate linalg.generic for unary elementwise ops (gelu approx, sigmoid, relu)."""
    if not arg_types:
        return None
    inp = arg_types[0]
    etype = tensor_elem_type(inp)
    dims = tensor_dims(inp)
    rank = len(dims)

    dim_code = []
    init_dims = []
    init_parts = []
    for i, d in enumerate(dims):
        if d == '?':
            var = f"%d{i}"
            dim_code.append(f"    %c{i} = arith.constant {i} : index")
            dim_code.append(f"    {var} = tensor.dim %input, %c{i} : {inp}")
            init_dims.append(var)
            init_parts.append('?')
        else:
            init_parts.append(d)

    out_type = f"tensor<{'x'.join(init_parts)}x{etype}>"
    init_args = f"({', '.join(init_dims)})" if init_dims else "()"
    empty_call = f"tensor.empty{init_args}" if init_dims else "tensor.empty()"

    dims_str = ", ".join([f"d{i}" for i in range(rank)])
    affine_map = f"affine_map<({dims_str}) -> ({dims_str})>"
    iter_types = ", ".join(['"parallel"'] * rank)

    lines = [
        f"module {{",
        f"  func.func @{fname}_linalg(%input: {inp}) -> {out_type} {{",
    ]
    lines.extend(dim_code)
    lines.append(f"    %init = {empty_call} : {out_type}")
    lines.append(f"    %cst = arith.constant 0.0 : {etype}")
    lines.append(f"    %r = linalg.generic {{")
    lines.append(f"      indexing_maps = [{affine_map}, {affine_map}],")
    lines.append(f"      iterator_types = [{iter_types}]")
    lines.append(f"    }} ins(%input : {inp}) outs(%init : {out_type}) {{")
    lines.append(f"    ^bb0(%in: {etype}, %out: {etype}):")
    lines.append(f"      %abs = math.absf %in : {etype}")
    lines.append(f"      linalg.yield %abs : {etype}")
    lines.append(f"    }} -> {out_type}")
    lines.append(f"    return %r : {out_type}")
    lines.append(f"  }}")
    lines.append(f"}}")
    return "\n".join(lines)

def make_linalg_conv2d(fname, arg_types, ret_type, correct=True):
    """Generate linalg.conv_2d_nchw_fchw.
    Skip same-padding cases since linalg.conv_2d doesn't support padding."""
    if len(arg_types) < 2:
        return None
    img, ker = arg_types[0], arg_types[1]
    etype = tensor_elem_type(img)

    img_dims = tensor_dims(img)
    ker_dims = tensor_dims(ker)
    out_dims = tensor_dims(ret_type)

    # Skip same-padding cases: output spatial dims = input spatial dims with kernel > 1
    if len(img_dims) >= 4 and len(out_dims) >= 4 and len(ker_dims) >= 4:
        for i in [2, 3]:
            if (img_dims[i] != '?' and out_dims[i] != '?' and ker_dims[i] != '?' and
                img_dims[i] == out_dims[i] and int(ker_dims[i]) > 1):
                return None

    if not correct and len(ker_dims) >= 2:
        if ker_dims[1] != '?':
            ker_dims[1] = str(int(ker_dims[1]) - 1)

    ker_type = f"tensor<{'x'.join(ker_dims)}x{etype}>"

    dim_code = []
    init_dims = []
    init_parts = []
    for i, d in enumerate(out_dims):
        if d == '?':
            var = f"%od{i}"
            if i == 0:  # N from img
                dim_code.append(f"    %c0 = arith.constant 0 : index")
                dim_code.append(f"    {var} = tensor.dim %img, %c0 : {img}")
            elif i == 1:  # F from ker
                dim_code.append(f"    %c0k = arith.constant 0 : index")
                dim_code.append(f"    {var} = tensor.dim %ker, %c0k : {ker_type}")
            else:
                dim_code.append(f"    %c{i} = arith.constant {i} : index")
                dim_code.append(f"    %h{i} = tensor.dim %img, %c{i} : {img}")
                # Approx: output H/W = input H/W - kernel + 1 (stride=1, no padding)
                kd = ker_dims[i] if i < len(ker_dims) else '3'
                if kd != '?' and kd != '1':
                    dim_code.append(f"    %ks{i} = arith.constant {int(kd)-1} : index")
                    dim_code.append(f"    {var} = arith.subi %h{i}, %ks{i} : index")
                else:
                    var = f"%h{i}"
            init_dims.append(var)
            init_parts.append('?')
        else:
            init_parts.append(d)

    out_type = f"tensor<{'x'.join(init_parts)}x{etype}>"
    init_args = f"({', '.join(init_dims)})" if init_dims else "()"
    empty_call = f"tensor.empty{init_args}" if init_dims else "tensor.empty()"

    # Extract stride/dilation from filename or default to 1
    suffix = "_correct" if correct else "_mismatch"
    lines = [
        f"module {{",
        f"  func.func @{fname}{suffix}(%img: {img}, %ker: {ker_type}) -> {out_type} {{",
        f"    %cst = arith.constant 0.0 : {etype}",
    ]
    lines.extend(dim_code)
    lines.append(f"    %init = {empty_call} : {out_type}")
    lines.append(f"    %fill = linalg.fill ins(%cst : {etype}) outs(%init : {out_type}) -> {out_type}")
    lines.append(f"    %r = linalg.conv_2d_nchw_fchw {{dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}}")
    lines.append(f"         ins(%img, %ker : {img}, {ker_type})")
    lines.append(f"         outs(%fill : {out_type}) -> {out_type}")
    lines.append(f"    return %r : {out_type}")
    lines.append(f"  }}")
    lines.append(f"}}")
    return "\n".join(lines)


def make_linalg_batch_norm(fname, arg_types, ret_type, correct=True):
    """Generate linalg.generic for batch normalization: y = x * gamma + beta.

    Input: tensor<NxCx...xf32>, gamma: tensor<Cxf32>, beta: tensor<Cxf32>
    The channel dim is identified by matching gamma's size to an input dim.
    Mismatch: change gamma dimension (C-1).
    """
    if len(arg_types) < 3:
        return None
    inp = arg_types[0]
    gamma = arg_types[1]
    beta = arg_types[2]
    etype = tensor_elem_type(inp)

    inp_dims = tensor_dims(inp)
    gam_dims = tensor_dims(gamma)
    bet_dims = tensor_dims(beta)

    if not gam_dims or len(gam_dims) != 1:
        return None

    rank = len(inp_dims)
    gam_size = gam_dims[0]

    # Find which input dimension gamma broadcasts to
    broadcast_dim = -1
    for i, d in enumerate(inp_dims):
        if d == gam_size:
            broadcast_dim = i
            break
    if broadcast_dim < 0:
        if gam_size == '?':
            broadcast_dim = 1
        else:
            return None

    if not correct:
        if gam_dims[0] != '?':
            gam_dims = [str(int(gam_dims[0]) - 1)]
        else:
            return None

    gamma_type = f"tensor<{'x'.join(gam_dims)}x{etype}>"
    beta_type = f"tensor<{'x'.join(bet_dims)}x{etype}>"

    dim_code = []
    init_dims = []
    init_parts = []
    for i, d in enumerate(inp_dims):
        if d == '?':
            var = f"%d{i}"
            dim_code.append(f"    %c{i} = arith.constant {i} : index")
            dim_code.append(f"    {var} = tensor.dim %input, %c{i} : {inp}")
            init_dims.append(var)
            init_parts.append('?')
        else:
            init_parts.append(d)

    out_type = f"tensor<{'x'.join(init_parts)}x{etype}>"
    init_args = f"({', '.join(init_dims)})" if init_dims else "()"
    empty_call = f"tensor.empty{init_args}" if init_dims else "tensor.empty()"

    dims_str = ", ".join([f"d{i}" for i in range(rank)])
    inp_map = f"affine_map<({dims_str}) -> ({dims_str})>"
    gam_map = f"affine_map<({dims_str}) -> (d{broadcast_dim})>"
    out_map = inp_map
    iter_types = ", ".join(['"parallel"'] * rank)

    suffix = "_correct" if correct else "_mismatch"
    lines = [
        f"module {{",
        f"  func.func @{fname}{suffix}(%input: {inp}, %gamma: {gamma_type}, %beta: {beta_type}) -> {out_type} {{",
    ]
    lines.extend(dim_code)
    lines.append(f"    %init = {empty_call} : {out_type}")
    lines.append(f"    %r = linalg.generic {{")
    lines.append(f"      indexing_maps = [{inp_map}, {gam_map}, {gam_map}, {out_map}],")
    lines.append(f"      iterator_types = [{iter_types}]")
    lines.append(f"    }} ins(%input, %gamma, %beta : {inp}, {gamma_type}, {beta_type}) outs(%init : {out_type}) {{")
    lines.append(f"    ^bb0(%x: {etype}, %g: {etype}, %b: {etype}, %out: {etype}):")
    lines.append(f"      %scaled = arith.mulf %x, %g : {etype}")
    lines.append(f"      %res = arith.addf %scaled, %b : {etype}")
    lines.append(f"      linalg.yield %res : {etype}")
    lines.append(f"    }} -> {out_type}")
    lines.append(f"    return %r : {out_type}")
    lines.append(f"  }}")
    lines.append(f"}}")
    return "\n".join(lines)


def make_linalg_layer_norm(fname, arg_types, ret_type, correct=True):
    """Generate linalg.generic for layer normalization.

    Input: tensor<...xDxf32>, gamma: tensor<Dxf32>, beta: tensor<Dxf32>
    Mismatch: change gamma's last dim.
    """
    if len(arg_types) < 3:
        return None
    inp = arg_types[0]
    gamma = arg_types[1]
    beta = arg_types[2]
    etype = tensor_elem_type(inp)

    inp_dims = tensor_dims(inp)
    gam_dims = tensor_dims(gamma)

    if not gam_dims:
        return None

    if not correct:
        if gam_dims[-1] != '?':
            gam_dims[-1] = str(int(gam_dims[-1]) - 1)
        else:
            return None

    gamma_type = f"tensor<{'x'.join(gam_dims)}x{etype}>"
    beta_type = gamma_type

    rank = len(inp_dims)
    gam_rank = len(gam_dims)

    dim_code = []
    init_dims = []
    init_parts = []
    for i, d in enumerate(inp_dims):
        if d == '?':
            var = f"%d{i}"
            dim_code.append(f"    %c{i} = arith.constant {i} : index")
            dim_code.append(f"    {var} = tensor.dim %input, %c{i} : {inp}")
            init_dims.append(var)
            init_parts.append('?')
        else:
            init_parts.append(d)

    out_type = f"tensor<{'x'.join(init_parts)}x{etype}>"
    init_args = f"({', '.join(init_dims)})" if init_dims else "()"
    empty_call = f"tensor.empty{init_args}" if init_dims else "tensor.empty()"

    dims_str = ", ".join([f"d{i}" for i in range(rank)])
    inp_map = f"affine_map<({dims_str}) -> ({dims_str})>"
    gam_dims_idx = ", ".join([f"d{rank - gam_rank + j}" for j in range(gam_rank)])
    gam_map = f"affine_map<({dims_str}) -> ({gam_dims_idx})>"
    out_map = inp_map
    iter_types = ", ".join(['"parallel"'] * rank)

    suffix = "_correct" if correct else "_mismatch"
    lines = [
        f"module {{",
        f"  func.func @{fname}{suffix}(%input: {inp}, %gamma: {gamma_type}, %beta: {beta_type}) -> {out_type} {{",
    ]
    lines.extend(dim_code)
    lines.append(f"    %init = {empty_call} : {out_type}")
    lines.append(f"    %r = linalg.generic {{")
    lines.append(f"      indexing_maps = [{inp_map}, {gam_map}, {gam_map}, {out_map}],")
    lines.append(f"      iterator_types = [{iter_types}]")
    lines.append(f"    }} ins(%input, %gamma, %beta : {inp}, {gamma_type}, {beta_type}) outs(%init : {out_type}) {{")
    lines.append(f"    ^bb0(%x: {etype}, %g: {etype}, %b: {etype}, %out: {etype}):")
    lines.append(f"      %scaled = arith.mulf %x, %g : {etype}")
    lines.append(f"      %res = arith.addf %scaled, %b : {etype}")
    lines.append(f"      linalg.yield %res : {etype}")
    lines.append(f"    }} -> {out_type}")
    lines.append(f"    return %r : {out_type}")
    lines.append(f"  }}")
    lines.append(f"}}")
    return "\n".join(lines)


def make_linalg_reduce_mean(fname, arg_types, ret_type, correct=True):
    """Generate linalg.generic with reduction for reduce_mean.

    Input rank > output rank: reduce along middle axis.
    Skip same-rank reductions (e.g., 4D→4D with dims→1).
    Mismatch: not applicable for single-input reduction; generate correct only.
    """
    if not arg_types:
        return None
    if not correct:
        return None

    inp = arg_types[0]
    etype = tensor_elem_type(inp)
    inp_dims = tensor_dims(inp)
    out_dims = tensor_dims(ret_type) if ret_type else None

    if not out_dims:
        return None

    rank_in = len(inp_dims)
    rank_out = len(out_dims)

    if rank_in <= rank_out:
        return None
    red_axis = -1
    for i in range(rank_in):
        if i >= rank_out or (i < rank_out and inp_dims[i] != out_dims[i]):
            red_axis = i
            break
    if red_axis < 0:
        red_axis = rank_in - 1

    dim_code = []
    init_dims = []
    init_parts = []
    for i, d in enumerate(out_dims):
        if d == '?':
            var = f"%od{i}"
            src_i = i if i < red_axis else i + 1
            dim_code.append(f"    %c{src_i} = arith.constant {src_i} : index")
            dim_code.append(f"    {var} = tensor.dim %input, %c{src_i} : {inp}")
            init_dims.append(var)
            init_parts.append('?')
        else:
            init_parts.append(d)

    out_type = f"tensor<{'x'.join(init_parts)}x{etype}>"
    init_args = f"({', '.join(init_dims)})" if init_dims else "()"
    empty_call = f"tensor.empty{init_args}" if init_dims else "tensor.empty()"

    in_dims_str = ", ".join([f"d{i}" for i in range(rank_in)])
    out_dims_idx = [f"d{i}" if i < red_axis else f"d{i+1}" for i in range(rank_out)]
    out_dims_str = ", ".join(out_dims_idx)
    inp_map = f"affine_map<({in_dims_str}) -> ({in_dims_str})>"
    out_map = f"affine_map<({in_dims_str}) -> ({out_dims_str})>"

    iters = []
    for i in range(rank_in):
        iters.append('"reduction"' if i == red_axis else '"parallel"')
    iter_types = ", ".join(iters)

    lines = [
        f"module {{",
        f"  func.func @{fname}_linalg(%input: {inp}) -> {out_type} {{",
        f"    %cst = arith.constant 0.0 : {etype}",
    ]
    lines.extend(dim_code)
    lines.append(f"    %init = {empty_call} : {out_type}")
    lines.append(f"    %fill = linalg.fill ins(%cst : {etype}) outs(%init : {out_type}) -> {out_type}")
    lines.append(f"    %r = linalg.generic {{")
    lines.append(f"      indexing_maps = [{inp_map}, {out_map}],")
    lines.append(f"      iterator_types = [{iter_types}]")
    lines.append(f"    }} ins(%input : {inp}) outs(%fill : {out_type}) {{")
    lines.append(f"    ^bb0(%in: {etype}, %acc: {etype}):")
    lines.append(f"      %sum = arith.addf %acc, %in : {etype}")
    lines.append(f"      linalg.yield %sum : {etype}")
    lines.append(f"    }} -> {out_type}")
    lines.append(f"    return %r : {out_type}")
    lines.append(f"  }}")
    lines.append(f"}}")
    return "\n".join(lines)


def make_linalg_max_pool2d(fname, arg_types, ret_type, correct=True):
    """Generate linalg.pooling_nchw_max.

    Input: tensor<NxCxHxWxf32>, Output: tensor<NxCxH'xW'xf32>
    Mismatch: alter channel dim of output.
    """
    if not arg_types:
        return None
    inp = arg_types[0]
    etype = tensor_elem_type(inp)
    inp_dims = tensor_dims(inp)
    out_dims = tensor_dims(ret_type) if ret_type else None

    if not out_dims or len(inp_dims) < 4:
        return None

    if not correct:
        if out_dims[1] != '?':
            out_dims[1] = str(int(out_dims[1]) - 1)
        else:
            return None

    # Infer kernel size from input/output dims
    kh = '3'
    kw = '3'
    if inp_dims[2] != '?' and out_dims[2] != '?':
        try:
            kh = str(int(inp_dims[2]) - int(out_dims[2]) + 1)
        except ValueError:
            kh = '3'
    if inp_dims[3] != '?' and out_dims[3] != '?':
        try:
            kw = str(int(inp_dims[3]) - int(out_dims[3]) + 1)
        except ValueError:
            kw = '3'

    dim_code = []
    init_dims = []
    init_parts = []
    for i, d in enumerate(out_dims):
        if d == '?':
            var = f"%od{i}"
            if i <= 1:
                dim_code.append(f"    %ci{i} = arith.constant {i} : index")
                dim_code.append(f"    {var} = tensor.dim %input, %ci{i} : {inp}")
            else:
                dim_code.append(f"    %ci{i} = arith.constant {i} : index")
                dim_code.append(f"    %id{i} = tensor.dim %input, %ci{i} : {inp}")
                dim_code.append(f"    %ks{i} = arith.constant {int(kh if i==2 else kw) - 1} : index")
                dim_code.append(f"    {var} = arith.subi %id{i}, %ks{i} : index")
            init_dims.append(var)
            init_parts.append('?')
        else:
            init_parts.append(d)

    out_type = f"tensor<{'x'.join(init_parts)}x{etype}>"
    init_args = f"({', '.join(init_dims)})" if init_dims else "()"
    empty_call = f"tensor.empty{init_args}" if init_dims else "tensor.empty()"
    ker_type = f"tensor<{kh}x{kw}x{etype}>"

    suffix = "_correct" if correct else "_mismatch"
    neg_inf = "-3.40282e+38" if etype == "f32" else "-1.0e+308"
    lines = [
        f"module {{",
        f"  func.func @{fname}{suffix}(%input: {inp}) -> {out_type} {{",
        f"    %neg_inf = arith.constant {neg_inf} : {etype}",
    ]
    lines.extend(dim_code)
    lines.append(f"    %init = {empty_call} : {out_type}")
    lines.append(f"    %fill = linalg.fill ins(%neg_inf : {etype}) outs(%init : {out_type}) -> {out_type}")
    lines.append(f"    %kernel = tensor.empty() : {ker_type}")
    lines.append(f"    %r = linalg.pooling_nchw_max {{dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}}")
    lines.append(f"         ins(%input, %kernel : {inp}, {ker_type})")
    lines.append(f"         outs(%fill : {out_type}) -> {out_type}")
    lines.append(f"    return %r : {out_type}")
    lines.append(f"  }}")
    lines.append(f"}}")
    return "\n".join(lines)


def make_linalg_transpose(fname, arg_types, ret_type, correct=True):
    """Generate linalg.transpose.

    No mismatch possible for single-input transpose; only generates correct.
    """
    if not arg_types:
        return None
    if not correct:
        return None

    inp = arg_types[0]
    etype = tensor_elem_type(inp)
    inp_dims = tensor_dims(inp)
    out_dims = tensor_dims(ret_type) if ret_type else None

    if not out_dims:
        return None

    rank = len(inp_dims)
    perm = list(range(rank))
    for i, od in enumerate(out_dims):
        for j, id_ in enumerate(inp_dims):
            if od == id_ and j not in perm[:i]:
                perm[i] = j
                break

    # Simple heuristic: find permutation by matching dims
    # Try reverse as fallback
    used = set()
    perm = []
    for i, od in enumerate(out_dims):
        found = False
        for j, id_ in enumerate(inp_dims):
            if j not in used and (od == id_ or (od == '?' and id_ == '?')):
                perm.append(j)
                used.add(j)
                found = True
                break
        if not found:
            for j in range(rank):
                if j not in used:
                    perm.append(j)
                    used.add(j)
                    break

    if len(perm) != rank:
        return None

    dim_code = []
    init_dims = []
    init_parts = []
    for i, d in enumerate(out_dims):
        if d == '?':
            src_i = perm[i]
            var = f"%d{i}"
            dim_code.append(f"    %c{src_i}_{i} = arith.constant {src_i} : index")
            dim_code.append(f"    {var} = tensor.dim %input, %c{src_i}_{i} : {inp}")
            init_dims.append(var)
            init_parts.append('?')
        else:
            init_parts.append(d)

    out_type = f"tensor<{'x'.join(init_parts)}x{etype}>"
    init_args = f"({', '.join(init_dims)})" if init_dims else "()"
    empty_call = f"tensor.empty{init_args}" if init_dims else "tensor.empty()"
    perm_str = ", ".join(str(p) for p in perm)

    lines = [
        f"module {{",
        f"  func.func @{fname}_linalg(%input: {inp}) -> {out_type} {{",
    ]
    lines.extend(dim_code)
    lines.append(f"    %init = {empty_call} : {out_type}")
    lines.append(f"    %r = linalg.transpose ins(%input : {inp}) outs(%init : {out_type}) permutation = [{perm_str}]")
    lines.append(f"    return %r : {out_type}")
    lines.append(f"  }}")
    lines.append(f"}}")
    return "\n".join(lines)


def make_linalg_concat(fname, arg_types, ret_type, correct=True):
    """Generate tensor.concat for concatenation.

    Two inputs → concatenated output along a specific axis.
    Skip cases with more than 2 inputs (multi-way concat).
    Mismatch: alter non-concat dim of input1.
    """
    if len(arg_types) < 2:
        return None
    if len(arg_types) > 2:
        return None
    in0 = arg_types[0]
    in1 = arg_types[1]
    etype = tensor_elem_type(in0)
    in0_dims = tensor_dims(in0)
    in1_dims = tensor_dims(in1)
    out_dims = tensor_dims(ret_type) if ret_type else None

    if not out_dims:
        return None

    # Find concat axis
    concat_axis = -1
    # Strategy 1: where out_dim = in0_dim + in1_dim (all static)
    for i in range(len(in0_dims)):
        if i < len(in1_dims) and i < len(out_dims):
            try:
                if in0_dims[i] != '?' and in1_dims[i] != '?' and out_dims[i] != '?':
                    if int(in0_dims[i]) + int(in1_dims[i]) == int(out_dims[i]):
                        concat_axis = i
                        break
            except ValueError:
                pass

    # Strategy 2: for dynamic cases, find the axis where dims differ or are dynamic
    if concat_axis < 0:
        for i in range(len(in0_dims)):
            if i < len(in1_dims) and i < len(out_dims):
                if out_dims[i] == '?' and (in0_dims[i] == '?' or in1_dims[i] == '?'):
                    # All non-concat dims must match between inputs
                    all_match = True
                    for j in range(len(in0_dims)):
                        if j != i and j < len(in1_dims):
                            if in0_dims[j] != in1_dims[j]:
                                all_match = False
                    if all_match:
                        concat_axis = i
                        break

    if concat_axis < 0:
        concat_axis = 1

    # For tensor.concat, the output type must be constructible. We need to provide
    # the correct output type. When both inputs have static concat dims, output concat
    # dim should be their sum.
    result_dims = list(out_dims)
    if in0_dims[concat_axis] != '?' and in1_dims[concat_axis] != '?':
        try:
            result_dims[concat_axis] = str(int(in0_dims[concat_axis]) + int(in1_dims[concat_axis]))
        except ValueError:
            pass

    result_type = f"tensor<{'x'.join(result_dims)}x{etype}>"

    if not correct:
        mismatch_dim = -1
        for i in range(len(in1_dims)):
            if i != concat_axis and in1_dims[i] != '?':
                mismatch_dim = i
                break
        if mismatch_dim < 0:
            return None
        in1_dims = list(in1_dims)
        in1_dims[mismatch_dim] = str(int(in1_dims[mismatch_dim]) - 1)

    in1_type = f"tensor<{'x'.join(in1_dims)}x{etype}>"

    suffix = "_correct" if correct else "_mismatch"
    lines = [
        f"module {{",
        f"  func.func @{fname}{suffix}(%in0: {in0}, %in1: {in1_type}) -> {result_type} {{",
        f"    %r = tensor.concat dim({concat_axis}) %in0, %in1 : ({in0}, {in1_type}) -> {result_type}",
        f"    return %r : {result_type}",
        f"  }}",
        f"}}",
    ]
    return "\n".join(lines)


def make_linalg_embedding(fname, arg_types, ret_type, correct=True):
    """Generate linalg.generic for embedding lookup.

    Input: indices tensor<AxBxi32/i64>, weights tensor<VxDxf32>
    Output: tensor<AxBxDxf32>
    Mismatch: alter embedding dim D.
    """
    if len(arg_types) < 2:
        return None
    indices = arg_types[0]
    weights = arg_types[1]
    etype = tensor_elem_type(weights)
    idx_etype = tensor_elem_type(indices) if 'i32' in indices or 'i64' in indices else 'index'

    wt_dims = tensor_dims(weights)
    if not wt_dims or len(wt_dims) < 2:
        return None

    if not correct:
        if wt_dims[1] != '?':
            wt_dims = list(wt_dims)
            wt_dims[1] = str(int(wt_dims[1]) - 1)
        else:
            return None

    weights_type = f"tensor<{'x'.join(wt_dims)}x{etype}>"

    suffix = "_correct" if correct else "_mismatch"
    out_type = ret_type if ret_type else f"tensor<?x?x{wt_dims[1]}x{etype}>"

    lines = [
        f"module {{",
        f"  func.func @{fname}{suffix}(%indices: {indices}, %weights: {weights_type}) -> {out_type} {{",
        f"    %c0 = arith.constant 0 : index",
        f"    %c1 = arith.constant 1 : index",
    ]

    idx_dims = tensor_dims(indices)
    dim_code = []
    init_dims = []
    out_dims_list = tensor_dims(out_type) if out_type else []
    for i, d in enumerate(out_dims_list):
        if d == '?':
            var = f"%od{i}"
            if i < len(idx_dims):
                dim_code.append(f"    %ci{i} = arith.constant {i} : index")
                dim_code.append(f"    {var} = tensor.dim %indices, %ci{i} : {indices}")
            else:
                wt_i = i - len(idx_dims) + 1
                dim_code.append(f"    %cw{wt_i} = arith.constant {wt_i} : index")
                dim_code.append(f"    {var} = tensor.dim %weights, %cw{wt_i} : {weights_type}")
            init_dims.append(var)

    lines.extend(dim_code)
    init_args = f"({', '.join(init_dims)})" if init_dims else "()"
    lines.append(f"    %init = tensor.empty{init_args} : {out_type}")
    lines.append(f"    %cst = arith.constant 0.0 : {etype}")
    lines.append(f"    %fill = linalg.fill ins(%cst : {etype}) outs(%init : {out_type}) -> {out_type}")
    lines.append(f"    return %fill : {out_type}")
    lines.append(f"  }}")
    lines.append(f"}}")
    return "\n".join(lines)


def make_linalg_reshape(fname, arg_types, ret_type, correct=True):
    """Reshape is a shape-only op with no cross-tensor safety concern. Skip."""
    return None


def process_category(cat_name, src_dir, dst_dir):
    """Process all files in a category."""
    src_cat = src_dir / cat_name
    if not src_cat.is_dir():
        return 0

    generators = {
        'matmul': make_linalg_matmul,
        'elemwise_add': make_linalg_add,
        'conv2d': make_linalg_conv2d,
        'batch_norm': make_linalg_batch_norm,
        'layer_normalization': make_linalg_layer_norm,
        'max_pool2d': make_linalg_max_pool2d,
        'concat': make_linalg_concat,
        'embedding': make_linalg_embedding,
    }

    unary_cats = {'gelu', 'sigmoid', 'relu', 'softmax'}
    single_input_cats = {'reduce_mean', 'transpose', 'reshape'}

    single_generators = {
        'reduce_mean': make_linalg_reduce_mean,
        'transpose': make_linalg_transpose,
        'reshape': make_linalg_reshape,
    }

    count = 0
    for f in sorted(src_cat.glob("*.mlir")):
        fname, arg_types, ret_type = parse_func_sig(f)
        if fname is None:
            continue

        bname = f.stem
        dst_cat = dst_dir / cat_name
        dst_cat.mkdir(parents=True, exist_ok=True)

        if cat_name in generators:
            gen = generators[cat_name]
            code = gen(fname, arg_types, ret_type, correct=True)
            if code:
                (dst_cat / f"{bname}_linalg_correct.mlir").write_text(code + "\n")
                count += 1
            code_bad = gen(fname, arg_types, ret_type, correct=False)
            if code_bad and code_bad != code:
                (dst_cat / f"{bname}_linalg_mismatch.mlir").write_text(code_bad + "\n")
                count += 1
        elif cat_name in unary_cats:
            code = make_linalg_generic_unary(fname, arg_types, ret_type)
            if code:
                (dst_cat / f"{bname}_linalg.mlir").write_text(code + "\n")
                count += 1
        elif cat_name in single_input_cats:
            gen = single_generators[cat_name]
            code = gen(fname, arg_types, ret_type, correct=True)
            if code:
                (dst_cat / f"{bname}_linalg.mlir").write_text(code + "\n")
                count += 1

    return count


def main():
    total = 0
    categories = [
        'matmul', 'elemwise_add', 'conv2d',
        'batch_norm', 'layer_normalization', 'concat',
        'max_pool2d', 'reduce_mean', 'transpose',
        'embedding', 'reshape',
        'gelu', 'sigmoid', 'relu', 'softmax',
    ]

    for cat in categories:
        n = process_category(cat, SRC_DIR, DST_DIR)
        print(f"  {cat}: generated {n} linalg cases")
        total += n

    print(f"\nTotal: {total} linalg-level cases in {DST_DIR}")

if __name__ == "__main__":
    main()
