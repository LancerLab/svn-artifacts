#!/usr/bin/env python3
"""Validation driver: compose, compile, run, check the oracle for all cases."""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compose as C
import emit as E
import mlirbench as B

CATS = ["matmul", "relu", "softmax", "transpose", "concat", "layer_normalization", "elemwise_add"]


def run_one(cat: str, size: str, dynamic: bool, work: str) -> str:
    case = C.make_case(cat, size=size, dynamic=dynamic)
    src = E.emit_kernel(case)
    tag = f"{cat}_{'dyn' if dynamic else 'static'}"
    mlir = os.path.join(work, tag + ".mlir")
    ll = os.path.join(work, tag + ".ll.mlir")
    with open(mlir, "w") as f:
        f.write(src)

    r = B.run_mlir_opt(B.pipeline("linalg", False), mlir, ll)
    if not r.ok:
        return f"{tag:28s} COMPILE-FAIL  {B._first_diagnostic(r.stdout + r.stderr)}"
    run = B.run_kernel(ll)
    ret = B._parse_return(run.stdout)
    if not run.ok:
        return f"{tag:28s} RUN-FAIL rc={run.rc} {B._first_diagnostic(run.stdout + run.stderr)}"
    status = "PASS" if ret == 0 else f"ORACLE-MISMATCH ret={ret}"
    return f"{tag:28s} {status}"


def main() -> int:
    print(B.describe_toolchain())
    with tempfile.TemporaryDirectory() as work:
        for size in ("small",):
            for cat in CATS:
                for dyn in (False, True):
                    print(run_one(cat, size, dyn, work))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
