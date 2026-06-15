#!/usr/bin/env python3
"""
Generate RUNNABLE linalg-level MLIR test cases for mlir-cpu-runner.

Each case has a @main() that allocates memrefs, runs linalg ops, and prints a result.
Generates both correct and mismatched versions (for runtime detection testing).

For each existing benchmark, extracts shapes from the function signature and generates:
  - Correct: matching dimensions, should produce correct result
  - Mismatch: reduced contraction/input dim (for dim-mismatch detection)

Usage: python3 scripts/generate_runnable_linalg.py
  Then test: scripts/run_linalg_suite.sh
"""

import re
import os
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC_DIR = WORKSPACE / "benchmark" / "mlir" / "cases"
DST_DIR = WORKSPACE / "benchmark" / "mlir" / "cases_linalg_run"


def parse_func_sig(filepath):
    """Extract tensor types from function signature."""
    with open(filepath) as f:
        content = f.read()
    m = re.search(r'func\.func\s+@(\w+)\(([^)]*)\)\s*->\s*(tensor<[^{]+)', content)
    if not m:
        return None, [], None
    fname = m.group(1)
    args_str = m.group(2)
    ret_type = m.group(3).strip().rstrip('{').strip()
    arg_types = []
    for arg in args_str.split(','):
        arg = arg.strip()
        tm = re.search(r'tensor<([^>]+)>', arg)
        if tm:
            arg_types.append(f"tensor<{tm.group(1)}>")
        elif re.search(r'\bf32\b', arg):
            arg_types.append('f32')
    return fname, arg_types, ret_type


def tensor_dims(ttype):
    """Return list of dimension strings."""
    m = re.search(r'tensor<(.+)>', ttype)
    if not m:
        return []
    inner = re.sub(r'x(f32|f64|i32|i64|index)$', '', m.group(1))
    return inner.split('x')


def tensor_elem_type(ttype):
    m = re.search(r'(f32|f64)', ttype)
    return m.group(1) if m else 'f32'


def resolve_dim(d, fallback=64):
    """Resolve a dimension: static int stays, '?' becomes fallback."""
    if d == '?':
        return fallback
    try:
        return int(d)
    except ValueError:
        return fallback


def gen_matmul_2d(fname, lhs_type, rhs_type, out_type, mismatch=False):
    """Generate runnable linalg.matmul for 2D cases."""
    etype = tensor_elem_type(lhs_type)
    dims_l = tensor_dims(lhs_type)
    dims_r = tensor_dims(rhs_type)
    dims_o = tensor_dims(out_type)

    M = resolve_dim(dims_l[0], 16)
    K = resolve_dim(dims_l[1] if len(dims_l) > 1 else '?', 256)
    N = resolve_dim(dims_r[1] if len(dims_r) > 1 else '?', 64)
    K_rhs = K if not mismatch else max(K - 1, 1)

    suffix = "correct" if not mismatch else "mismatch"
    return f"""module {{
  func.func @main() {{
    %cM = arith.constant {M} : index
    %cK = arith.constant {K} : index
    %cK_rhs = arith.constant {K_rhs} : index
    %cN = arith.constant {N} : index
    %cst = arith.constant 0.0 : {etype}
    %cst1 = arith.constant 1.0 : {etype}

    %lhs = memref.alloc(%cM, %cK) : memref<?x?x{etype}>
    %rhs = memref.alloc(%cK_rhs, %cN) : memref<?x?x{etype}>
    %out = memref.alloc(%cM, %cN) : memref<?x?x{etype}>

    linalg.fill ins(%cst1 : {etype}) outs(%lhs : memref<?x?x{etype}>)
    linalg.fill ins(%cst1 : {etype}) outs(%rhs : memref<?x?x{etype}>)
    linalg.fill ins(%cst : {etype}) outs(%out : memref<?x?x{etype}>)

    linalg.matmul ins(%lhs, %rhs : memref<?x?x{etype}>, memref<?x?x{etype}>)
                   outs(%out : memref<?x?x{etype}>)

    %c0 = arith.constant 0 : index
    %val = memref.load %out[%c0, %c0] : memref<?x?x{etype}>
    vector.print %val : {etype}

    memref.dealloc %lhs : memref<?x?x{etype}>
    memref.dealloc %rhs : memref<?x?x{etype}>
    memref.dealloc %out : memref<?x?x{etype}>
    return
  }}
}}
"""


def gen_matmul_3d2d(fname, lhs_type, rhs_type, out_type, mismatch=False):
    """Generate runnable linalg.generic for 3D×2D broadcast matmul."""
    etype = tensor_elem_type(lhs_type)
    dims_l = tensor_dims(lhs_type)
    dims_r = tensor_dims(rhs_type)

    B = resolve_dim(dims_l[0], 2)
    M = resolve_dim(dims_l[1], 16)
    K = resolve_dim(dims_l[2] if len(dims_l) > 2 else '?', 256)
    N = resolve_dim(dims_r[1] if len(dims_r) > 1 else '?', 64)
    K_rhs = K if not mismatch else max(K - 1, 1)

    return f"""#map_lhs = affine_map<(b, m, n, k) -> (b, m, k)>
#map_rhs = affine_map<(b, m, n, k) -> (k, n)>
#map_out = affine_map<(b, m, n, k) -> (b, m, n)>

module {{
  func.func @main() {{
    %cB = arith.constant {B} : index
    %cM = arith.constant {M} : index
    %cK = arith.constant {K} : index
    %cK_rhs = arith.constant {K_rhs} : index
    %cN = arith.constant {N} : index
    %cst = arith.constant 0.0 : {etype}
    %cst1 = arith.constant 1.0 : {etype}

    %lhs = memref.alloc(%cB, %cM, %cK) : memref<?x?x?x{etype}>
    %rhs = memref.alloc(%cK_rhs, %cN) : memref<?x?x{etype}>
    %out = memref.alloc(%cB, %cM, %cN) : memref<?x?x?x{etype}>

    linalg.fill ins(%cst1 : {etype}) outs(%lhs : memref<?x?x?x{etype}>)
    linalg.fill ins(%cst1 : {etype}) outs(%rhs : memref<?x?x{etype}>)
    linalg.fill ins(%cst : {etype}) outs(%out : memref<?x?x?x{etype}>)

    linalg.generic {{
      indexing_maps = [#map_lhs, #map_rhs, #map_out],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    }} ins(%lhs, %rhs : memref<?x?x?x{etype}>, memref<?x?x{etype}>)
      outs(%out : memref<?x?x?x{etype}>) {{
    ^bb0(%a: {etype}, %b: {etype}, %c: {etype}):
      %mul = arith.mulf %a, %b : {etype}
      %add = arith.addf %c, %mul : {etype}
      linalg.yield %add : {etype}
    }}

    %c0 = arith.constant 0 : index
    %val = memref.load %out[%c0, %c0, %c0] : memref<?x?x?x{etype}>
    vector.print %val : {etype}

    memref.dealloc %lhs : memref<?x?x?x{etype}>
    memref.dealloc %rhs : memref<?x?x{etype}>
    memref.dealloc %out : memref<?x?x?x{etype}>
    return
  }}
}}
"""


def gen_add(fname, lhs_type, rhs_type, out_type, mismatch=False):
    """Generate runnable linalg.add."""
    etype = tensor_elem_type(lhs_type)
    dims = tensor_dims(lhs_type)
    dims_r = tensor_dims(rhs_type)

    # Check if rhs is broadcast (fewer dims or scalar)
    if len(dims_r) < len(dims):
        return None  # Skip broadcast cases for now

    resolved = [resolve_dim(d, 16) for d in dims]
    resolved_r = list(resolved)
    if mismatch:
        for i in range(len(resolved_r)):
            if resolved_r[i] > 1:
                resolved_r[i] = max(resolved_r[i] - 1, 1)
                break

    rank = len(resolved)
    alloc_dims = ", ".join(f"%c{i}" for i in range(rank))
    alloc_r_dims = ", ".join(f"%cr{i}" for i in range(rank))
    mem_type = f"memref<{'x'.join(['?']*rank)}x{etype}>"

    lines = [f"module {{", f"  func.func @main() {{"]
    for i, d in enumerate(resolved):
        lines.append(f"    %c{i} = arith.constant {d} : index")
    for i, d in enumerate(resolved_r):
        lines.append(f"    %cr{i} = arith.constant {d} : index")
    lines.append(f"    %cst1 = arith.constant 1.0 : {etype}")
    lines.append(f"    %cst2 = arith.constant 2.0 : {etype}")
    lines.append(f"    %a = memref.alloc({alloc_dims}) : {mem_type}")
    lines.append(f"    %b = memref.alloc({alloc_r_dims}) : {mem_type}")
    lines.append(f"    %out = memref.alloc({alloc_dims}) : {mem_type}")
    lines.append(f"    linalg.fill ins(%cst1 : {etype}) outs(%a : {mem_type})")
    lines.append(f"    linalg.fill ins(%cst2 : {etype}) outs(%b : {mem_type})")
    lines.append(f"    linalg.add ins(%a, %b : {mem_type}, {mem_type})")
    lines.append(f"              outs(%out : {mem_type})")
    zero_idx = ", ".join([f"%z{i}" for i in range(rank)])
    for i in range(rank):
        lines.append(f"    %z{i} = arith.constant 0 : index")
    lines.append(f"    %val = memref.load %out[{zero_idx}] : {mem_type}")
    lines.append(f"    vector.print %val : {etype}")
    lines.append(f"    memref.dealloc %a : {mem_type}")
    lines.append(f"    memref.dealloc %b : {mem_type}")
    lines.append(f"    memref.dealloc %out : {mem_type}")
    lines.append(f"    return")
    lines.append(f"  }}")
    lines.append(f"}}")
    return "\n".join(lines)


def gen_unary(fname, inp_type, out_type):
    """Generate runnable linalg.generic for unary elementwise op."""
    etype = tensor_elem_type(inp_type)
    dims = tensor_dims(inp_type)
    resolved = [resolve_dim(d, 16) for d in dims]
    rank = len(resolved)

    alloc_dims = ", ".join(f"%c{i}" for i in range(rank))
    mem_type = f"memref<{'x'.join(['?']*rank)}x{etype}>"
    dims_str = ", ".join([f"d{i}" for i in range(rank)])
    amap = f"affine_map<({dims_str}) -> ({dims_str})>"
    iter_types = ", ".join(['"parallel"'] * rank)
    zero_idx = ", ".join([f"%z{i}" for i in range(rank)])

    lines = [
        f"#map = {amap}",
        f"module {{",
        f"  func.func @main() {{",
    ]
    for i, d in enumerate(resolved):
        lines.append(f"    %c{i} = arith.constant {d} : index")
    lines.append(f"    %cst_neg = arith.constant -2.5 : {etype}")
    lines.append(f"    %inp = memref.alloc({alloc_dims}) : {mem_type}")
    lines.append(f"    %out = memref.alloc({alloc_dims}) : {mem_type}")
    lines.append(f"    linalg.fill ins(%cst_neg : {etype}) outs(%inp : {mem_type})")
    lines.append(f"    linalg.generic {{")
    lines.append(f"      indexing_maps = [#map, #map],")
    lines.append(f"      iterator_types = [{iter_types}]")
    lines.append(f"    }} ins(%inp : {mem_type}) outs(%out : {mem_type}) {{")
    lines.append(f"    ^bb0(%in: {etype}, %unused: {etype}):")
    lines.append(f"      %abs = math.absf %in : {etype}")
    lines.append(f"      linalg.yield %abs : {etype}")
    lines.append(f"    }}")
    for i in range(rank):
        lines.append(f"    %z{i} = arith.constant 0 : index")
    lines.append(f"    %val = memref.load %out[{zero_idx}] : {mem_type}")
    lines.append(f"    vector.print %val : {etype}")
    lines.append(f"    memref.dealloc %inp : {mem_type}")
    lines.append(f"    memref.dealloc %out : {mem_type}")
    lines.append(f"    return")
    lines.append(f"  }}")
    lines.append(f"}}")
    return "\n".join(lines)


def process_all():
    total = 0

    for cat_name in ['matmul', 'elemwise_add', 'conv2d', 'gelu', 'sigmoid', 'relu', 'softmax']:
        src_cat = SRC_DIR / cat_name
        if not src_cat.is_dir():
            continue

        dst_cat = DST_DIR / cat_name
        dst_cat.mkdir(parents=True, exist_ok=True)
        cat_count = 0

        for f in sorted(src_cat.glob("*.mlir")):
            fname, arg_types, ret_type = parse_func_sig(f)
            if fname is None or not arg_types:
                continue

            bname = f.stem

            if cat_name in ('gelu', 'sigmoid', 'relu', 'softmax'):
                code = gen_unary(fname, arg_types[0], ret_type)
                if code:
                    (dst_cat / f"{bname}_run.mlir").write_text(code + "\n")
                    cat_count += 1
            elif cat_name == 'matmul':
                lhs, rhs = arg_types[0], arg_types[1] if len(arg_types) > 1 else arg_types[0]
                rank_l = len(tensor_dims(lhs))
                rank_r = len(tensor_dims(rhs))

                if rank_l == 2 and rank_r == 2:
                    code = gen_matmul_2d(fname, lhs, rhs, ret_type, mismatch=False)
                    code_bad = gen_matmul_2d(fname, lhs, rhs, ret_type, mismatch=True)
                elif rank_l == 3 and rank_r == 2:
                    code = gen_matmul_3d2d(fname, lhs, rhs, ret_type, mismatch=False)
                    code_bad = gen_matmul_3d2d(fname, lhs, rhs, ret_type, mismatch=True)
                elif rank_l == 4 and rank_r == 4:
                    continue  # Skip 4D attention for now
                else:
                    continue

                if code:
                    (dst_cat / f"{bname}_run_correct.mlir").write_text(code + "\n")
                    cat_count += 1
                if code_bad:
                    (dst_cat / f"{bname}_run_mismatch.mlir").write_text(code_bad + "\n")
                    cat_count += 1
            elif cat_name == 'elemwise_add':
                if len(arg_types) < 2:
                    continue
                code = gen_add(fname, arg_types[0], arg_types[1], ret_type, mismatch=False)
                code_bad = gen_add(fname, arg_types[0], arg_types[1], ret_type, mismatch=True)
                if code:
                    (dst_cat / f"{bname}_run_correct.mlir").write_text(code + "\n")
                    cat_count += 1
                if code_bad:
                    (dst_cat / f"{bname}_run_mismatch.mlir").write_text(code_bad + "\n")
                    cat_count += 1

        print(f"  {cat_name}: generated {cat_count} runnable cases")
        total += cat_count

    print(f"\nTotal: {total} runnable linalg cases in {DST_DIR}")


if __name__ == "__main__":
    process_all()
