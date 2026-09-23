#!/usr/bin/env python3
"""lane.py — CUTLASS/CuTe candidate lane driver for benchmark2.

Vertical slice: for the spec-required operator set it (1) builds and gates
unmutated kernels against the CPU references in reference.py, then (2) applies
the realizable mutation battery and classifies each mutant using the §9.6
outcome taxonomy:

    ct-check   compile fails
    rt-check   run faults (CUT_CHECK / non-zero exit)
    noop       output byte-identical to base  -> discarded (n_discarded_noop)
    unchecked  output differs, no diagnostic   -> silent corruption

Descriptor/TMA families (M3.2-M3.16) are emitted by probes.py, not here.

Usage:
    lane.py base     --out raw/
    lane.py mutants  --out raw/ --records records.jsonl
    lane.py all
    lane.py collect  --records records.jsonl --out results/
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
CUDA = "/usr/local/cuda-12.9/bin/nvcc"
CUTLASS = os.path.abspath(os.path.join(
    ROOT, "..", "..", "croqtile", "extern", "cutlass"))
ARCH = "sm_86"
INCLUDES = ["-I", os.path.join(CUTLASS, "include"),
            "-I", os.path.join(CUTLASS, "tools", "util", "include"),
            "-I", os.path.join(ROOT, "kernels")]

sys.path.insert(0, ROOT)
import reference as R          # noqa: E402
import sizes                   # noqa: E402
import mutrec                  # noqa: E402

# spec-required operators for M3/M4 (mutation-specs-v2 §5/§6)
M3_OPS = ["matmul", "conv2d", "batch_norm", "max_pool2d"]
M4_OPS = ["layer_normalization", "softmax", "matmul", "elemwise_add"]
# remaining shape-bearing categories (coverage extension, not spec-required)
EXTRA_OPS = ["relu", "sigmoid", "gelu", "reshape", "transpose", "concat",
             "embedding", "reduce_mean"]
OPS = sorted(set(M3_OPS) | set(M4_OPS) | set(EXTRA_OPS))

SRC = {"elemwise_add": "kernels/elemwise_add.cu", "relu": "kernels/relu.cu",
       "sigmoid": "kernels/sigmoid.cu", "gelu": "kernels/gelu.cu",
       "reshape": "kernels/reshape.cu", "transpose": "kernels/transpose.cu",
       "concat": "kernels/concat.cu", "embedding": "kernels/embedding.cu",
       "reduce_mean": "kernels/reduce_mean.cu",
       "softmax": "kernels/softmax.cu",
       "layer_normalization": "kernels/layer_normalization.cu",
       "matmul": "kernels/matmul.cu", "conv2d": "kernels/conv2d.cu",
       "batch_norm": "kernels/batch_norm.cu", "max_pool2d": "kernels/max_pool2d.cu"}


def shape_flags(cat, shape):
    if cat in ("elemwise_add", "relu", "sigmoid", "gelu", "reshape"):
        return {"N_ELEM": shape[0]}
    if cat == "transpose":
        return {"TR_M": shape[0], "TR_N": shape[1]}
    if cat == "concat":
        return {"CC_NA": shape[0], "CC_NB": shape[1]}
    if cat == "embedding":
        return {"EM_V": shape[0], "EM_D": shape[1], "EM_N": shape[2]}
    if cat == "reduce_mean":
        return {"RM_ROWS": shape[0], "RM_COLS": shape[1]}
    if cat in ("softmax", "layer_normalization"):
        return {"ROWS": shape[0], "COLS": shape[1]}
    if cat == "matmul":
        return {"MM": shape[0], "KK": shape[1], "NN": shape[2]}
    if cat == "conv2d":
        B, C, H, W, CO, KH, KW = shape
        return {"BB": B, "CC": C, "HH": H, "WW": W, "COO": CO,
                "KHh": KH, "KWw": KW}
    if cat == "batch_norm":
        N, C, H, W = shape
        return {"NN": N, "CC": C, "HH": H, "WW": W}
    if cat == "max_pool2d":
        C, H, W, K = shape
        return {"Ch": C, "HH": H, "WW": W, "KK": K}
    raise KeyError(cat)


def build(cat, shape, mut, outbin):
    flags = dict(shape_flags(cat, shape))
    flags.update({k: str(v) for k, v in (mut or {}).items()})
    cmd = [CUDA, "-std=c++17", f"-arch={ARCH}", "-O2"] + INCLUDES + \
          [f"-D{k}={v}" for k, v in flags.items()] + [SRC[cat], "-o", outbin]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stderr


def run_bin(binpath, outpath, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run([binpath, outpath], capture_output=True, text=True, env=e)
    return p.returncode, p.stderr


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def src_hash(cat):
    with open(os.path.join(ROOT, SRC[cat]), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def settings_hash(cat, shape):
    blob = json.dumps({"cat": cat, "shape": list(shape), "arch": ARCH},
                      sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def classify_record(spec_id, cat, mut, outcome, detail, kernel_hash,
                    settings_hash_v):
    """Map the lane's §9.6 outcome onto a v2.1 record (see mutrec.py)."""
    tag = f"{cat}.{spec_id.replace('.', '_')}." + \
          "_".join(f"{k}{v}" for k, v in sorted((mut or {}).items()))
    mutid = f"cutlass-{tag}"
    common = dict(spec_id=spec_id, category=cat, mutation=mut, outcome=outcome,
                  mutant_id=mutid, detail=detail, kernel_hash=kernel_hash,
                  settings_hash=settings_hash_v, arch=ARCH)
    if spec_id.startswith("L"):
        # launch-status path class: the limit is held by the driver, so the
        # lane observes it rather than detaining it (applicable=false).
        manifest = "noop" if outcome == "noop" else "undecidable"
        return mutrec.make_record(path_class="L", manifest=manifest,
                                  prohibition="observation", applicable=False,
                                  **common)
    if outcome == "ct-check":
        return mutrec.make_record(path_class="ct-check", manifest="value-changing",
                                  prohibition="", applicable=True, **common)
    if outcome == "rt-check":
        return mutrec.make_record(path_class="rt-check", manifest="value-changing",
                                  prohibition="absent", applicable=True,
                                  **common)
    if outcome == "unchecked":
        # M3 is compile-only here: the defect compiled, but we did not run it,
        # so whether it corrupts is undecidable. M4 is launched and the output
        # differs, so it corrupts.
        manifest = "undecidable" if spec_id.startswith("M3") else "value-changing"
        return mutrec.make_record(path_class="unchecked", manifest=manifest,
                                  prohibition="absent", applicable=True,
                                  **common)
    if outcome == "noop":
        return mutrec.make_record(path_class="avoided", manifest="noop",
                                  prohibition="absent", applicable=False,
                                  **common)
    raise ValueError(f"unknown outcome {outcome!r}")


def base_pass(outdir):
    """Build+gate every operator; return {cat: {"bin","out","shape"}}."""
    os.makedirs(outdir, exist_ok=True)
    base = {}
    for cat in OPS:
        shape = sizes.shape(cat)
        binp = os.path.join(outdir, f"{cat}.base")
        outp = os.path.join(outdir, f"{cat}.base.bin")
        rc, err = build(cat, shape, {}, binp)
        if rc != 0:
            raise RuntimeError(f"[{cat}] base compile failed:\n{err[-2000:]}")
        rc, err = run_bin(binp, outp)
        if rc != 0:
            raise RuntimeError(f"[{cat}] base run failed rc={rc}:\n{err[-2000:]}")
        got = R.read_bin(outp)
        ref = R.reference(cat, shape, R.inputs(cat, shape))
        ok, mad, mrd = R.gate(cat, got, ref)
        if not ok:
            raise RuntimeError(f"[{cat}] base GATE failed max_abs={mad:.3e}")
        base[cat] = {"bin": binp, "out": outp, "shape": shape,
                     "sha": sha(outp), "max_abs": mad}
        print(f"[base] {cat:22s} OK  n={got.size:7d}  max_abs={mad:.2e}")
    return base


# --- realizable mutation battery (vertical slice) -------------------------
# Descriptor/TMA/atom specs are type-level; see probes.py.
# Each entry: (spec_id, category, compile_macros, runtime_env)
BATTERY = [
    ("M4.1", "elemwise_add", {"LOOP": 0}, None),
    ("M4.1", "softmax", {"LOOP": 0}, None),
    ("M4.1", "layer_normalization", {"LOOP": 0}, None),
    ("M4.1", "matmul", {"LOOP": 0}, None),
    ("M4.1", "conv2d", {"LOOP": 0}, None),
    ("M4.1", "max_pool2d", {"LOOP": 0}, None),
    ("M4.1", "batch_norm", {"LOOP": 0}, None),
    ("M4.1", "relu", {"LOOP": 0}, None),
    ("M4.1", "sigmoid", {"LOOP": 0}, None),
    ("M4.1", "gelu", {"LOOP": 0}, None),
    ("M4.1", "reshape", {"LOOP": 0}, None),
    ("M4.1", "transpose", {"LOOP": 0}, None),
    ("M4.1", "concat", {"LOOP": 0}, None),
    ("M4.1", "embedding", {"LOOP": 0}, None),
    ("M4.1", "reduce_mean", {"LOOP": 0}, None),
    ("M4.3", "elemwise_add", {}, {"CUT_LOOP_RT": "0"}),
    ("M4.3", "softmax", {}, {"CUT_LOOP_RT": "0"}),
    ("M4.3", "layer_normalization", {}, {"CUT_LOOP_RT": "0"}),
    ("M4.3", "matmul", {}, {"CUT_LOOP_RT": "0"}),
    ("M4.5", "matmul", {"STEP": 0}, None),
    ("M4.5", "conv2d", {"STEP": 0}, None),
    ("L1", "elemwise_add", {"SMEM_ELT": 32768}, None),
    ("L1", "matmul", {"SMEM_ELT": 32768}, None),
]


def mutants_pass(base, outdir, records_path):
    recs = []
    kh = {cat: src_hash(cat) for cat in SRC}
    sh = {cat: settings_hash(cat, base[cat]["shape"]) for cat in SRC}
    for spec_id, cat, mut, env in BATTERY:
        shape = base[cat]["shape"]
        allmut = dict(mut or {}, **(env or {}))
        tag = f"{cat}.{spec_id.replace('.', '_')}." + \
              "_".join(f"{k}{v}" for k, v in sorted((mut or {}).items()))
        if env:
            tag += "." + "_".join(f"{k}{v}" for k, v in sorted(env.items()))
        binp = os.path.join(outdir, tag)
        outp = os.path.join(outdir, tag + ".bin")
        rc, err = build(cat, shape, mut, binp)
        if rc != 0:
            r = classify_record(spec_id, cat, allmut, "ct-check",
                                err.strip().split("\n")[-1], kh[cat], sh[cat])
        else:
            rc2, err2 = run_bin(binp, outp, env)
            if rc2 != 0:
                # §9.6.1b rule 3: only an *emitted* check (CUT_CHECK) is an
                # rt-check; any other non-zero exit is a bare crash -> never.
                if "CUT_CHECK" in err2:
                    r = classify_record(spec_id, cat, allmut, "rt-check",
                                        err2.strip()[-200:], kh[cat], sh[cat])
                else:
                    r = classify_record(spec_id, cat, allmut, "unchecked",
                                        "raw crash: " + err2.strip()[-180:],
                                        kh[cat], sh[cat])
            elif not os.path.exists(outp) or sha(outp) == base[cat]["sha"]:
                r = classify_record(spec_id, cat, allmut, "noop", "",
                                    kh[cat], sh[cat])
            else:
                got = R.read_bin(outp)
                r = classify_record(spec_id, cat, allmut, "unchecked",
                                    f"n={got.size} vs base n={int(os.path.getsize(base[cat]['out']) // 4)}",
                                    kh[cat], sh[cat])
        recs.append(r)
        print(f"[mut] {spec_id:6s} {cat:22s} -> {r['outcome']:8s} "
              f"{r['path_class']:9s} {r['detail']}")
    if records_path:
        with open(records_path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["base", "mutants", "all", "collect"])
    ap.add_argument("--out", default=os.path.join(ROOT, "raw"))
    ap.add_argument("--records", default=os.path.join(ROOT, "records.jsonl"))
    ap.add_argument("--collect-out", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    if args.cmd in ("base", "all"):
        base_pass(args.out)
    if args.cmd in ("mutants", "all"):
        base = {}
        for cat in OPS:
            outp = os.path.join(args.out, f"{cat}.base.bin")
            base[cat] = {"out": outp, "shape": sizes.shape(cat), "sha": sha(outp)}
        mutants_pass(base, args.out, args.records)
    if args.cmd == "collect":
        import collect
        collect.main(args.records, args.collect_out)


if __name__ == "__main__":
    main()
