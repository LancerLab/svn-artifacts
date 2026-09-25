#!/usr/bin/env python3
"""Shared harness for the benchmark2 MLIR lanes (`mlir-linalg`, `mlir-low`).

Both lanes compose their own kernels from `benchmark2/settings/`, mutate them per
`benchmark2/specs/mutation-specs.md`, and classify every mutant into the
`{compile, runtime, never, n/a}` taxonomy with the §7 ground-truth oracle.

This module owns everything that must be *identical* across the two lanes so the
numbers are comparable:

* the pinned LLVM-21 toolchain location and the exact pass pipelines,
* assert counting (which is stage-dependent -- see `count_asserts`),
* the compile / run / classify protocol,
* the §7 manifest oracle (`value-changing` vs `noop`),
* record emission against `schema/record-schema.json`.

Measurement rules that are easy to get wrong and are therefore enforced here:

1. **Assert counting depends on the IR stage.** `cf.assert` exists only at MLIR
   level; after full LLVM lowering it becomes `assert_msg_N` globals plus `abort`
   calls. Counting `cf.assert` in lowered IR silently returns 0. Use
   `count_asserts`, which picks the right pattern for the stage.
2. **`stdbuf -o0` is mandatory** when running: the assert message is written to
   stdout via `puts` before `abort()` and is otherwise lost to buffering.
3. **A missing shared library gives exit 1** (JIT symbol error), which is trivially
   misread as a detection. `run_kernel` distinguishes that case explicitly.
4. **RTV placement is a correctness requirement.** It instruments `memref.load` /
   `tensor.extract` but *not* `affine.load`, so `lower-affine` must run before it.
   The pipelines below encode the validated orderings.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

# `schema/records.py` is the ONE record validator (see its module docstring for
# why there used to be four). This file reaches it rather than re-deriving the
# rule, because the rule is not "the fields listed under `fields`": a record's
# required set depends on its RELEASE, and a v1 record must stay valid.
_B2 = Path(__file__).resolve().parent.parent                   # benchmark2/
if str(_B2) not in sys.path:
    sys.path.insert(0, str(_B2))
from schema import records as _records                         # noqa: E402

# --------------------------------------------------------------------------
# Toolchain (pinned in benchmark2/manifest.md §5.1)
# --------------------------------------------------------------------------

def _resolve_llvm_root() -> Path:
    """Locate the pinned LLVM/MLIR 21.1.0 tree.

    `MLIR_LLVM_ROOT` wins when set. Otherwise try, in order: the in-repo
    croqtile `extern/` tree (`svn-artifacts/croqtile/extern/llvm-project`, the
    one the committed `raw/setup.json` records and the one a clean checkout
    actually has), then the historical reference-host paths. The first
    candidate that carries `bin/mlir-opt` is used; if none exists the in-repo
    path is returned so the toolchain check reports it rather than an unrelated
    absolute path.
    """
    env = os.environ.get("MLIR_LLVM_ROOT")
    if env:
        return Path(env).resolve()
    candidates = [
        _B2.parent / "croqtile" / "extern" / "llvm-project",
        Path("/home/garfee/dev/croqtile/extern/llvm-project"),
        Path.home() / "dev" / "croqtile" / "extern" / "llvm-project",
    ]
    for c in candidates:
        if (c / "bin" / "mlir-opt").exists():
            return c.resolve()
    return candidates[0].resolve()


LLVM_ROOT = _resolve_llvm_root()
LLVM_BIN = LLVM_ROOT / "bin"
LLVM_LIB = LLVM_ROOT / "lib"

MLIR_OPT = LLVM_BIN / "mlir-opt"
# `mlir-cpu-runner` was renamed `mlir-runner` in LLVM 21.
MLIR_RUNNER = LLVM_BIN / "mlir-runner"
# S12 (sanitizer supplement) lowers to LLVM IR and builds a *native* ASan binary.
# These three are the reason the MLIR lanes can report a real S12 measurement
# instead of n/a; `check_toolchain()` verifies they exist.
MLIR_TRANSLATE = LLVM_BIN / "mlir-translate"
CLANG = LLVM_BIN / "clang"
LLVM_OPT = LLVM_BIN / "opt"

RUNNER_LIBS = ",".join(
    str(LLVM_LIB / n) for n in ("libmlir_runner_utils.so", "libmlir_c_runner_utils.so")
)

# The GPU path needs the CUDA runtime in addition to the two CPU runner libs:
# it is the shared object that owns `mgpuLaunchKernel`/`mgpuMemAlloc`, which
# `gpu-to-llvm` emits calls to. It ships with MLIR but is only built when
# `MLIR_ENABLE_CUDA_RUNNER=ON` (`llvm-project/lib/libmlir_cuda_runtime.so`).
CUDA_RUNTIME_LIB = LLVM_LIB / "libmlir_cuda_runtime.so"
RUNNER_LIBS_GPU = RUNNER_LIBS + "," + str(CUDA_RUNTIME_LIB)

# S12 on the GPU surface uses `compute-sanitizer` (the CUDA memcheck) instead of
# ASan: the kernel runs on the device, where a natively-linked ASan binary
# cannot observe the access. Pinned by absolute path because it ships with the
# CUDA toolkit and is not on `PATH`; `check_toolchain()` verifies it.
COMPUTE_SANITIZER = Path(
    os.environ.get("MLIR_COMPUTE_SANITIZER", "/usr/local/cuda/bin/compute-sanitizer")
)
SANITIZER_EXIT = 99  # the `--error-exitcode` passed below

TOOLCHAIN_VERSION = "LLVM 21.1.0"

# Exit codes we care about.
EXIT_ABORT = 134  # assert fired -> abort()
EXIT_SEGV = 139
EXIT_JIT_SYMBOL = 1  # missing shared libs / unresolved symbol -- NOT a detection
# Synthetic: the kernel never terminated within `timeout`. Distinct from every
# real exit status so a hang is classified on its own merits rather than falling
# into the generic "nonzero rc" bucket. An OOB write can corrupt the heap in a way
# that loops forever instead of faulting, so this is a genuine detection outcome
# for M1, not a harness failure.
EXIT_TIMEOUT = -1

_JIT_SYMBOL_MARKERS = (
    "JIT session error",
    "Symbols not found",
    "Failed to materialize symbols",
)

# --------------------------------------------------------------------------
# Pass pipelines (validated on LLVM 21.1.0)
# --------------------------------------------------------------------------
#
# Pass nesting matters: `finalize-memref-to-llvm`, `convert-cf-to-llvm`,
# `convert-func-to-llvm`, `convert-arith-to-llvm`, `convert-scf-to-cf`,
# `convert-vector-to-llvm`, `one-shot-bufferize` and
# `buffer-deallocation-pipeline` are module-level and must sit directly under
# `builtin.module(...)`. `lower-affine`, `convert-linalg-to-loops` and
# `generate-runtime-verification` are func-level.

# Tail shared by every pipeline: MLIR dialects -> LLVM.
#
# `convert-math-to-llvm` is required because the composed kernels use
# `math.exp` (softmax) and `math.sqrt` (layer_norm).
# `expand-strided-metadata` is required because `tensor.concat` bufferizes to
# `memref.subview` with a strided layout that `finalize-memref-to-llvm` will not
# convert on its own.
_TAIL = (
    "convert-scf-to-cf,"
    "convert-vector-to-llvm,"
    "convert-math-to-llvm,"
    "convert-index-to-llvm,"  # required: RTV emits index.bool.constant
    "expand-strided-metadata,"
    "convert-arith-to-llvm,"
    "finalize-memref-to-llvm,"
    "convert-cf-to-llvm,"
    "convert-func-to-llvm,"
    "reconcile-unrealized-casts"
)

# Low-surface variant: expand-strided-metadata materializes an `affine.apply` for
# the linear address of a strided view (`memref.subview`, M1.20), and the runner
# refuses IR that still carries the affine dialect. A second `lower-affine`
# immediately after the expansion lowers it. The pass adds no checks, so S1/S9
# are untouched, and it is a no-op for every kernel without a strided view -- the
# linalg surface keeps the shared `_TAIL` because changing it would move that
# lane's already-recorded measurements.
_TAIL_LOW = _TAIL.replace(
    "expand-strided-metadata,", "expand-strided-metadata,func.func(lower-affine),"
)

# --- mlir-linalg: tensor-level linalg, needs bufferization -----------------
_LINALG_BODY_OFF = (
    "func.func(lower-affine),"
    "one-shot-bufferize,"
    "buffer-deallocation-pipeline,"
    "func.func(convert-linalg-to-loops)"
)
# RTV placement is NOT a free choice -- it is forced, and it scales S9 by ~4x, so
# the reasoning is recorded here rather than left implicit.
#
# Two candidate placements were measured on the relu kernel:
#
#   (A) bufferize -> convert-linalg-to-loops -> lower-affine -> RTV   [PINNED]
#       RTV sees memref.load/memref.store. 5 cf.assert, all
#       "^ out-of-bounds access": dynamic memory guards. Lowers to LLVM cleanly
#       and runs (exit 0).
#
#   (B) convert-linalg-to-loops -> RTV, with NO bufferization
#       20 cf.assert, but they are "^ dimension #N ... incompatible" and
#       "^ unexpected negative result on dimension #N": *static* shape checks,
#       not memory guards. convert-linalg-to-loops is a no-op on tensor ops, so
#       12 linalg ops survive to the LLVM tail, which then fails with
#       "Dialect `tensor' not found for custom op 'tensor.empty'". The kernel
#       cannot be built, let alone run.
#
# (B) is therefore not a valid configuration at all, and its higher count is not
# a real measurement of instrumentation cost. (A) is pinned because it is the only
# placement that yields an executable kernel. lower-affine must sit between
# convert-linalg-to-loops and RTV because linalg->loops emits affine.load.
_LINALG_BODY_ON = (
    "func.func(lower-affine),"
    "one-shot-bufferize,"
    "buffer-deallocation-pipeline,"
    "func.func(convert-linalg-to-loops,lower-affine,generate-runtime-verification)"
)

PIPELINES = {
    ("linalg", False): f"builtin.module({_LINALG_BODY_OFF},{_TAIL})",
    ("linalg", True): f"builtin.module({_LINALG_BODY_ON},{_TAIL})",
    # --- mlir-low: already on memref, no bufferization --------------------
    ("low", False): f"builtin.module(func.func(lower-affine),{_TAIL_LOW})",
    ("low", True): (
        "builtin.module(func.func(lower-affine,generate-runtime-verification,"
        f"canonicalize,cse),{_TAIL_LOW})"
    ),
}

# Pipeline used only to count how many checks RTV generated, before lowering.
RTV_ONLY = {
    "linalg": (
        "builtin.module(func.func(lower-affine),one-shot-bufferize,"
        "buffer-deallocation-pipeline,"
        "func.func(convert-linalg-to-loops,lower-affine,"
        "generate-runtime-verification))"
    ),
    "low": "builtin.module(func.func(lower-affine,generate-runtime-verification))",
}


def pipeline(surface: str, rtv: bool) -> str:
    """Return the pinned pass pipeline for a surface and RTV setting."""
    return PIPELINES[(surface, rtv)]


# --------------------------------------------------------------------------
# GPU path (mlir-linalg on CUDA)
# --------------------------------------------------------------------------
#
# The lane audits *GPU* kernels, so the linalg surface executes on the device.
# The pipelines below are the GPU counterpart of PIPELINES' linalg entries: the
# same bufferize/lower-affine front end, but linalg maps to hardware instead of
# to `scf` loops, and the device module is serialized to a cubin.
#
# RTV sits after `lower-affine` and before the GPU mapping, so its `cf.assert`
# guards land *inside* the outlined `gpu.func` and keep their `loc("kernel")`
# tag. That is what lets the S9 census separate the kernel's guards from the host
# oracle's, exactly as on the CPU path (verified: kernel guards appear in
# `gpu.func`, oracle guards stay in the host function).
#
# The host oracle stays on the CPU by construction: a linalg body with reduction
# iterators does not map to `gpu.thread_id`, so `convert-linalg-to-parallel-loops`
# only turns the parallel kernel body into `scf.parallel`; the reduction lowers
# to host `scf.for`. Only the parallel body is outlined into a `gpu.module`. The
# host/device split is therefore a property of the pipeline, not a hand-written
# boundary in the emitter.
_GPU_MID_BODY = (
    "func.func(lower-affine),"
    "one-shot-bufferize,"
    "buffer-deallocation-pipeline,"
    "func.func(convert-linalg-to-parallel-loops,lower-affine"
)
_GPU_MID_TAIL = (
    ",gpu-map-parallel-loops,convert-parallel-loops-to-gpu),"
    "gpu-kernel-outlining"
)

GPU_MID = {
    ("linalg", False): f"builtin.module({_GPU_MID_BODY}{_GPU_MID_TAIL})",
    ("linalg", True): (
        f"builtin.module({_GPU_MID_BODY},generate-runtime-verification"
        f"{_GPU_MID_TAIL})"
    ),
}

# Guard census for the GPU surface runs the same RTV-on mid pipeline, so the
# counted `cf.assert`s are the ones that actually reach the device module.
GPU_RTV_ONLY = {
    ("linalg", True): GPU_MID[("linalg", True)],
    # The low surface already carries its own `gpu.launch` in the source, so RTV
    # instruments the kernel body in place: the one-step host pipeline is also
    # the pre-GPU IR, and its `loc("kernel")` asserts are exactly the ones that
    # lower into the device module.
    ("low", True): RTV_ONLY["low"],
}

# mlir-low GPU front end. The source is memref/affine on the host plus an
# explicit `gpu.launch` for the kernel body, so the only work before the NVVM
# pipeline is lowering the *host* affine (the checksum oracle). Letting the
# NVVM pipeline do it does not work: GPUToNVVM runs `scf-to-cf` before
# `lower-affine`, so a surviving host `affine.for` is converted to `scf.for` and
# left un-lowered. Device memory is `gpu.alloc`-backed, so the linalg path's
# `gpu.host_register` injection is neither needed nor wanted.
GPU_LOW_PRE = {
    False: "builtin.module(func.func(lower-affine))",
    True: (
        "builtin.module(func.func(lower-affine,generate-runtime-verification,"
        "canonicalize,cse))"
    ),
}


def cuda_chip() -> str:
    """Target SM for cubin serialization.

    Host-parametric: the dev box is `sm_86` while the final host may be `sm_120`.
    `MLIR_CUDA_CHIP` overrides; otherwise the driver is asked. A wrong chip makes
    every compile fail with an opaque ptxas error, so the value is never guessed
    silently.
    """
    env = os.environ.get("MLIR_CUDA_CHIP")
    if env:
        return env
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        if out and out[0].strip():
            return "sm_" + out[0].strip().replace(".", "")
    except Exception:
        pass
    return "sm_86"


def backend_for(surface: str) -> str:
    """Execution backend for a surface: `cuda` for linalg/low, else `cpu`.

    The env overrides (`MLIR_LINALG_BACKEND`, `MLIR_LOW_BACKEND`) keep the
    previously committed CPU numbers reproducible without a code edit.
    """
    if surface == "linalg":
        return "cpu" if os.environ.get("MLIR_LINALG_BACKEND", "cuda").lower() == "cpu" else "cuda"
    if surface == "low":
        return "cpu" if os.environ.get("MLIR_LOW_BACKEND", "cuda").lower() == "cpu" else "cuda"
    return "cpu"


# --------------------------------------------------------------------------
# Host-buffer registration for the GPU path
# --------------------------------------------------------------------------
#
# `gpu-to-llvm` passes kernel memrefs through as raw pointers. A buffer from
# `memref.alloc` is host memory, so the driver faults on the first device access
# (`CUDA_ERROR_ILLEGAL_ADDRESS`) unless it is registered first. `gpu.host_register`
# is the op that performs `cuMemHostRegister`; no upstream pass inserts it, so the
# GPU path does it here.
#
# Only memrefs that are actually passed to a `gpu.launch_func` are registered,
# deduplicated by SSA name. Registering *every* allocation is wrong: the small
# scalar accumulators used by the host oracle trip `cuMemHostRegister` with
# `CUDA_ERROR_NOT_SUPPORTED`, and a value passed to two launches must be
# registered once.
_HOSTREG_ARG_RE = re.compile(r"(%[\w.]+)\s*:\s*(memref<[^>]*>)")
_HOSTREG_DEF_RE = re.compile(r"^(\s*)(%[\w.]+)\s*=")


def _unranked_memref(ty: str) -> str:
    """`memref<2x3xf32>` -> `memref<*xf32>` (the type `gpu.host_register` takes)."""
    return "memref<*x" + ty[len("memref<"):-1].split("x")[-1] + ">"


def insert_host_register(text: str) -> str:
    """Register every memref passed to a `gpu.launch_func`, after its definition.

    Element types in this harness are flat (`f32`/`i32`/`i64`), so the shape
    prefix is everything before the last `x` of the memref body.
    """
    needed: dict[str, str] = {}
    for line in text.splitlines():
        if "gpu.launch_func" in line:
            for name, ty in _HOSTREG_ARG_RE.findall(line):
                needed[name] = ty
    if not needed:
        return text
    out: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        out.append(line)
        m = _HOSTREG_DEF_RE.match(line)
        if m and m.group(2) in needed and m.group(2) not in seen:
            name = m.group(2)
            ty = needed[name]
            seen.add(name)
            ind, ur, urt = m.group(1), name + "_hostreg", _unranked_memref(ty)
            out.append(f"{ind}{ur} = memref.cast {name} : {ty} to {urt}")
            out.append(f"{ind}gpu.host_register {ur} : {urt}")
    return "\n".join(out) + "\n"


def lower_gpu(src: Path, work: Path, surface: str, rtv: bool, ll: Path) -> ProcResult:
    """Two-step GPU lowering.

    For `linalg`, step one maps linalg to `parallel` then `gpu`, and step two
    serializes the outlined device module to a cubin. For `low`, the source
    already holds the `gpu.launch`, so step one only lowers the host affine (with
    RTV if requested) and step two runs the NVVM pipeline.

    Returns the first failing `ProcResult`, or the final (successful) one. The
    intermediate IR is written beside `ll` so a failing mutant leaves the IR that
    produced the diagnostic on disk for inspection.
    """
    if surface == "low":
        pre = work / f"{src.stem}.gpu.pre.mlir"
        res = run_mlir_opt(GPU_LOW_PRE[rtv], src, pre)
        if not res.ok:
            return res
        return _run([
            str(MLIR_OPT), str(pre),
            "-gpu-lower-to-nvvm-pipeline="
            f"cubin-format=bin cubin-chip={cuda_chip()} opt-level=3",
            "-o", str(ll),
        ])

    mid = work / f"{src.stem}.gpu.mid.mlir"
    res = run_mlir_opt(GPU_MID[(surface, rtv)], src, mid)
    if not res.ok:
        return res
    reg = work / f"{src.stem}.gpu.hostreg.mlir"
    reg.write_text(insert_host_register(mid.read_text()))
    return _run([
        str(MLIR_OPT), str(reg),
        "-gpu-lower-to-nvvm-pipeline="
        f"cubin-format=bin cubin-chip={cuda_chip()} opt-level=3",
        # The upstream pipeline lowers memrefs only where the GPU conversion
        # touches them, so a host-side `memref.reshape` (the dynamic-reshape
        # oracle) survives into an otherwise-LLVM module that `mlir-runner`
        # refuses to parse. The CPU path handles it via full bufferization; here
        # the residual memref op is finalized explicitly. No-op when none remain.
        "-finalize-memref-to-llvm",
        "-reconcile-unrealized-casts",
        "-o", str(ll),
    ])


# --------------------------------------------------------------------------
# Process helpers
# --------------------------------------------------------------------------


@dataclass
class ProcResult:
    rc: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.rc == 0


def _as_text(v) -> str:
    """Normalize captured output to str.

    Even with `text=True`, `subprocess.TimeoutExpired` carries `stdout`/`stderr`
    as **bytes** -- a long-standing CPython quirk. Concatenating a str marker onto
    them raises `TypeError: can't concat str to bytes`, which would turn a hung
    kernel into a harness crash. Out-of-bounds mutants hang often enough (an OOB
    *write* corrupts the heap, and the damage may not surface until a later
    allocation) that this path is routinely hit, not theoretical.
    """
    if v is None:
        return ""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return v


def _run(cmd: list[str], timeout: int = 300) -> ProcResult:
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return ProcResult(p.returncode, p.stdout, p.stderr)
    except subprocess.TimeoutExpired as e:
        return ProcResult(
            EXIT_TIMEOUT,
            _as_text(e.stdout),
            _as_text(e.stderr) + f"\n[mlirbench] TIMEOUT after {timeout}s",
        )
    except FileNotFoundError as e:
        return ProcResult(-2, "", f"[mlirbench] binary not found: {e}")


def _run_pg(cmd: list[str], timeout: int = 300,
            env: dict | None = None) -> ProcResult:
    """Run `cmd` in its own session, killing the WHOLE process group on timeout.

    `subprocess.run(timeout=...)` kills only the direct child. The GPU sanitizer
    is `compute-sanitizer`, which re-parents the `mlir-runner` grandchild under a
    `TreeLauncherSubreaper`; a timed-out run -- the M4.5 zero-step mutants never
    terminate -- therefore leaves an orphan holding GPU memory that the next
    lane (choreo's exclusive timing run) cannot reclaim. A new session plus
    `killpg` reaps the tree.
    """
    import signal
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=env, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
        return ProcResult(p.returncode, out or "", err or "")
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            out, err = p.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        return ProcResult(
            EXIT_TIMEOUT, out or "",
            (err or "") + f"\n[mlirbench] TIMEOUT after {timeout}s (group killed)",
        )


def run_mlir_opt(pipeline_str: str, src: Path, dst: Path | None = None) -> ProcResult:
    """Run `mlir-opt` with an explicit pass pipeline.

    `dst=None` means "parse/verify only" (output discarded), which is how the
    verifier gate is measured.
    """
    cmd = [str(MLIR_OPT), f"--pass-pipeline={pipeline_str}", str(src)]
    if dst is not None:
        cmd += ["-o", str(dst)]
    else:
        cmd += ["-o", os.devnull]
    return _run(cmd)


def verify(src: Path) -> ProcResult:
    """Bare verifier gate: no passes, just parse + verify."""
    return _run([str(MLIR_OPT), str(src), "-o", os.devnull])


def run_kernel(
    ll_mlir: Path, entry_result: str = "i32", timeout: int = 120, gpu: bool = False
) -> ProcResult:
    """Execute lowered IR with `mlir-runner`.

    `stdbuf -o0` is mandatory: the RTV assert message goes to stdout via `puts`
    before `abort()` and is lost to buffering otherwise.

    `gpu=True` loads the CUDA runtime alongside the CPU runner libs, so a module
    that calls `mgpuLaunchKernel` can resolve it.
    """
    cmd = [
        "stdbuf",
        "-o0",
        str(MLIR_RUNNER),
        f"--entry-point-result={entry_result}",
        f"--shared-libs={RUNNER_LIBS_GPU if gpu else RUNNER_LIBS}",
        str(ll_mlir),
    ]
    return _run(cmd, timeout=timeout)


# --------------------------------------------------------------------------
# AddressSanitizer (S12) -- a SEPARATE execution path from the JIT
# --------------------------------------------------------------------------
#
# S1 measures what the toolchain itself catches, via `mlir-runner` (JIT). S12
# measures what an external memory checker catches on the *same* mutants, via a
# native ASan binary. The two are different execution models and must never be
# merged into one number: a mutant can be silent under the JIT and still be
# flagged by ASan, which is exactly the gap S12 exists to quantify.
#
# Three defects make the naive route (`mlir-translate | clang -fsanitize=address`)
# silently produce an UNINSTRUMENTED binary, so every one of them is handled here.
# Each was confirmed by measurement, not assumed:
#
#  1. clang does not instrument `.ll` inputs. Feeding it IR from a plain C file
#     compiled with `-emit-llvm` yields zero access checks; the binary runs and
#     reports nothing. Instrumentation must be done by `opt -passes=asan`.
#  2. LLVM's ASan function pass only instruments functions carrying the
#     `sanitize_address` attribute. `mlir-translate` never emits it, so the pass
#     runs, registers globals, and instruments nothing. The attribute is added
#     to every `define` below.
#  3. `mlir-translate` emits no `target triple`/`target datalayout`. Without them
#     ASan computes the wrong shadow-memory offset and a genuine overflow
#     surfaces as `SEGV on unknown address 0x1dc1...` instead of the
#     `heap-buffer-overflow` it actually is. Both are prepended, derived from the
#     host via clang rather than hardcoded.
#
# ASAN_INSTRUMENTED is the sanity gate for all of the above: it counts the
# `__asan_report_{load,store}N` call sites `opt` inserts on each error path. If it
# is 0 the binary was not instrumented, and any "clean" verdict is a FALSE
# NEGATIVE -- so callers must treat 0 as a harness failure, not as a result.

ASAN_ENV = {
    # Harness kernels `memref.alloc` -> `malloc` and never free; that is by
    # design, and LeakSanitizer would otherwise fire first and mask the fault
    # class we actually care about.
    "ASAN_OPTIONS": "detect_leaks=0:exitcode=86:abort_on_error=0",
    "LSAN_OPTIONS": "detect_leaks=0",
}
ASAN_EXIT = 86  # the `exitcode` set above; distinguishes an ASan report from a crash


def _asan_run_env() -> dict:
    """Environment for executing a native ASan binary.

    The runner utility libraries are linked in (see `run_asan`) but live outside
    the default loader search path, so without `LD_LIBRARY_PATH` the binary dies
    with exit 127 -- which would be misread as "the kernel crashed" rather than
    "the harness could not start it".
    """
    env = {**os.environ, **ASAN_ENV}
    env["LD_LIBRARY_PATH"] = str(LLVM_LIB) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env

_DEFINE_RE = re.compile(r"^(define [^{]*)\{", re.M)
_ASAN_CHECK_RE = re.compile(r"call void @__asan_report_(?:load|store)(?:\d+|N)\b")
_ASAN_ERROR_RE = re.compile(r"ERROR: AddressSanitizer: (\S+)")
_ASAN_LOCATED_RE = re.compile(r"is located (\d+) bytes (after|before|inside)")

# ASan fault class -> the schema's `fault` enum {oob, uninit, misaligned, none}.
# ASan cannot detect uninitialized reads (that is MemorySanitizer), so `uninit`
# is unreachable on this lane and is never emitted.
_ASAN_FAULT_CLASS = {
    "heap-buffer-overflow": "oob",
    "stack-buffer-overflow": "oob",
    "global-buffer-overflow": "oob",
    "heap-use-after-free": "oob",
    "SEGV": "oob",
    "alignment": "misaligned",
    "odr-violation": "none",
}


@dataclass
class AsanResult:
    """Outcome of one native ASan run.

    `instrumented` is the false-negative guard: if 0, the binary carried no
    access checks and `flagged`/`fault` are meaningless.
    """

    ok: bool  # the whole lower->instrument->build->run chain completed
    stage: str  # where it failed if not ok: lower/translate/opt/clang/run
    instrumented: int  # __asan_report_{load,store}N call sites in the IR
    flagged: bool  # ASan reported a fault
    fault: str  # schema enum: oob | uninit | misaligned | none
    exercised: bool  # the kernel body actually ran (see below)
    rc: int
    detail: str

    def as_record(self, toolchain: str, category: str, klass: str, mutant_id: str) -> dict:
        rec = {
            "toolchain": toolchain,
            "category": category,
            "class": klass,
            "mutant_id": mutant_id,
            "flagged": str(self.flagged).lower(),
            "fault": self.fault,
            "exercised": str(self.exercised).lower(),
            "toolchain_version": TOOLCHAIN_VERSION,
        }
        if self.detail:
            rec["detail"] = self.detail[:160]
        return rec


def _host_target_lines() -> tuple[str, str]:
    """Derive `target datalayout`/`target triple` from the host clang.

    Hardcoding x86_64 would silently break on any other host, and a wrong
    datalayout produces a wrong shadow offset -- faults reported as SEGV rather
    than as the overflow they are. Asking clang is both portable and exact.
    """
    triple = _run([str(CLANG), "-print-target-triple"]).stdout.strip()
    if not triple:
        triple = "x86_64-unknown-linux-gnu"
    # An empty translation unit carries the canonical datalayout for this host.
    with tempfile.TemporaryDirectory() as td:
        c = Path(td) / "t.c"
        c.write_text("int main(void){return 0;}\n")
        r = _run([str(CLANG), "-S", "-emit-llvm", "-O0", str(c), "-o", "-"])
        dl = ""
        for line in r.stdout.splitlines():
            if line.startswith("target datalayout"):
                dl = line.strip()
                break
    if not dl:
        dl = (
            'target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64'
            '-i64:64-i128:128-f80:128-n8:16:32:64-S128"'
        )
    return dl, f'target triple = "{triple}"'


_TARGET_LINES: tuple[str, str] | None = None


def target_lines() -> tuple[str, str]:
    global _TARGET_LINES
    if _TARGET_LINES is None:
        _TARGET_LINES = _host_target_lines()
    return _TARGET_LINES


def run_asan(
    src_mlir: Path,
    work: Path,
    surface: str,
    rtv: bool = False,
    timeout: int = 120,
) -> AsanResult:
    """Lower `src_mlir`, instrument with ASan, build natively, run, parse.

    `rtv=False` is the right choice for S12: RTV inserts its own `cf.assert`
    bounds checks, which would abort *before* the faulty access and make ASan
    silent. S12 asks what an external checker catches on the bare pipeline, so
    the same RTV-off lowering S1 uses for its "bare MLIR" baseline is used here.
    """
    work.mkdir(parents=True, exist_ok=True)
    ll = work / "asan.ll.mlir"
    ir = work / "asan.ll"
    patched = work / "asan.patched.ll"
    inst = work / "asan.inst.ll"
    binary = work / "asan.bin"

    r = run_mlir_opt(pipeline(surface, rtv), src_mlir, ll)
    if not r.ok:
        return AsanResult(False, "lower", 0, False, "none", False, r.rc,
                          _first_diagnostic(r.stderr) or "mlir-opt failed")

    t = _run([str(MLIR_TRANSLATE), "--mlir-to-llvmir", str(ll), "-o", str(ir)])
    if not t.ok:
        return AsanResult(False, "translate", 0, False, "none", False, t.rc,
                          _first_diagnostic(t.stderr) or "mlir-translate failed")

    # Defect 2: add `sanitize_address` to every defined function.
    # Defect 3: prepend the host triple/datalayout.
    dl, tt = target_lines()
    text = ir.read_text(errors="replace")
    text = _DEFINE_RE.sub(lambda m: m.group(1).rstrip() + " sanitize_address {", text)
    patched.write_text(f"{dl}\n{tt}\n{text}")

    o = _run([str(LLVM_OPT), "-passes=asan", str(patched), "-o", str(inst), "-S"])
    if not o.ok:
        return AsanResult(False, "opt", 0, False, "none", False, o.rc,
                          _first_diagnostic(o.stderr) or "opt -passes=asan failed")

    instrumented = len(_ASAN_CHECK_RE.findall(inst.read_text(errors="replace")))
    if instrumented == 0:
        # Not a measurement -- the binary would run unchecked and report clean.
        return AsanResult(False, "instrument", 0, False, "none", False, 0,
                          "opt inserted no __asan_report_* checks")

    # The runner utility libraries must be linked explicitly. Under the JIT,
    # `mlir-runner --shared-libs=` resolves helpers such as `memrefCopy` at run
    # time; a native link has no such mechanism and fails with
    # "undefined reference to `memrefCopy'" on any kernel that copies a memref.
    c = _run(
        [str(CLANG), "-fsanitize=address", str(inst), "-o", str(binary)]
        + RUNNER_LIBS.split(",")
    )
    if not c.ok:
        return AsanResult(False, "clang", instrumented, False, "none", False, c.rc,
                          _first_diagnostic(c.stderr) or "clang link failed")

    # `_run` takes no `env`, and both ASAN_OPTIONS and LD_LIBRARY_PATH must be in
    # force, so this one call is made explicitly.
    try:
        p = subprocess.run(
            [str(binary)], capture_output=True, text=True,
            timeout=timeout, env=_asan_run_env(),
        )
        run = ProcResult(p.returncode, p.stdout, p.stderr)
    except subprocess.TimeoutExpired as e:
        run = ProcResult(EXIT_TIMEOUT, _as_text(e.stdout), _as_text(e.stderr))
    except FileNotFoundError as e:
        return AsanResult(False, "run", instrumented, False, "none", False, -2,
                          f"asan binary not found: {e}")

    if run.rc == 127:
        # Loader failure, not a kernel outcome -- do not let it masquerade as a
        # crash and silently zero out `exercised`.
        return AsanResult(False, "run", instrumented, False, "none", False, 127,
                          "shared library load failed (rc=127)")

    err = run.stderr + run.stdout
    m = _ASAN_ERROR_RE.search(err)
    if m:
        klass = m.group(1)
        fault = _ASAN_FAULT_CLASS.get(klass, "oob")
        # Where the faulting address sits relative to the allocation is the
        # strongest evidence that the *injected* defect fired, so keep it.
        loc = _ASAN_LOCATED_RE.search(err)
        detail = f"asan: {klass}"
        if loc:
            detail += f" ({loc.group(1)}B {loc.group(2)})"
        # The sanitizer observed the faulting access itself, so the body ran.
        return AsanResult(True, "run", instrumented, True, fault, True,
                          run.rc, detail)

    # No ASan report. `exercised` follows the triton convention: did the kernel
    # body actually run? In this native path `main` returns the oracle's
    # *mismatch count*, so a small non-negative rc is a NORMAL completion, not a
    # failure -- rc=1 means "ran to the end, one value disagreed". Only a crash
    # or a hang means the body did not complete. Getting this backwards would
    # mark every wrong-result mutant as unexercised and understate S12.
    crashed = run.rc in (EXIT_ABORT, EXIT_SEGV, EXIT_TIMEOUT) or run.rc < 0
    exercised = not crashed
    if exercised:
        # Ran clean to completion: a genuine sanitizer negative. M1.6
        # (zero-stride/empty-range) lands here on every category -- it performs
        # no out-of-bounds access at all, so it is a defect class ASan
        # structurally cannot catch. That gap is the point of S12, not a bug.
        detail = "clean"
    else:
        detail = f"asan: no report, body did not complete (rc={run.rc})"
    return AsanResult(True, "run", instrumented, False, "none", exercised,
                      run.rc, detail)


_CUDA_ERROR_SUMMARY_RE = re.compile(r"ERROR SUMMARY: (\d+) errors?")
_CUDA_OOB_RE = re.compile(r"Invalid __global__ (read|write)")
_CUDA_MISALIGN_RE = re.compile(r"Misaligned address")


def run_sanitizer_gpu(
    src_mlir: Path,
    work: Path,
    surface: str,
    rtv: bool = False,
    timeout: int = 300,
) -> AsanResult:
    """S12 on the GPU surface: run the lowered cubin under `compute-sanitizer`.

    The interface matches `run_asan` so the caller's tally/`as_record` code does
    not fork. The `instrumented` gate changes meaning, though: on the native path
    it counts ASan call sites, while here it counts the device launch the
    sanitizer will actually watch. Zero launches means there was no device work
    to check, and a "clean" verdict would be a false negative -- the same failure
    mode the ASan path guards against.

    `rtv=False` for the same reason as `run_asan`: RTV's generated `cf.assert`
    fires on the device *before* the faulty access, so memcheck would see a
    clean run. S12 asks what an external checker catches on the bare pipeline.
    """
    work.mkdir(parents=True, exist_ok=True)
    ll = work / "sanitizer.gpu.mlir"
    res = lower_gpu(src_mlir, work, surface, rtv, ll)
    if not res.ok:
        return AsanResult(False, "lower", 0, False, "none", False, res.rc,
                          _first_diagnostic(res.stderr or res.stdout)
                          or "GPU lowering failed")

    instrumented = len(re.findall(r"\bgpu\.launch_func\b",
                                  ll.read_text(errors="replace")))
    if instrumented == 0:
        return AsanResult(False, "instrument", 0, False, "none", False, 0,
                          "no device launch in the lowered module")

    cmd = [
        str(COMPUTE_SANITIZER), "--tool", "memcheck",
        "--error-exitcode", str(SANITIZER_EXIT),
        str(MLIR_RUNNER), "--entry-point-result=i32",
        f"--shared-libs={RUNNER_LIBS_GPU}", str(ll),
    ]
    env = {**os.environ}
    env["LD_LIBRARY_PATH"] = str(LLVM_LIB) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    try:
        # Group-kill on timeout: compute-sanitizer's grandchild survives a plain
        # child kill and would leak GPU memory (see `_run_pg`).
        run = _run_pg(cmd, timeout=timeout, env=env)
    except FileNotFoundError as e:
        return AsanResult(False, "run", instrumented, False, "none", False, -2,
                          f"compute-sanitizer not found: {e}")

    err = run.stderr + run.stdout
    summary = _CUDA_ERROR_SUMMARY_RE.search(err)
    n_err = int(summary.group(1)) if summary else 0
    if n_err > 0 or run.rc == SANITIZER_EXIT:
        fault = "oob"
        cls = "CUDA memcheck"
        m = _CUDA_OOB_RE.search(err)
        if m:
            cls = f"Invalid __global__ {m.group(1)}"
        elif _CUDA_MISALIGN_RE.search(err):
            fault = "misaligned"
            cls = "Misaligned address"
        detail = f"compute-sanitizer: {cls}"
        if summary:
            detail += f" (ERROR SUMMARY: {n_err})"
        return AsanResult(True, "run", instrumented, True, fault, True,
                          run.rc, detail)

    crashed = run.rc in (EXIT_ABORT, EXIT_SEGV, EXIT_TIMEOUT) or run.rc < 0
    exercised = not crashed
    detail = ("clean" if exercised
              else f"compute-sanitizer: no report, body did not complete "
                   f"(rc={run.rc})")
    return AsanResult(True, "run", instrumented, False, "none", exercised,
                      run.rc, detail)


def run_sanitizer(
    src_mlir: Path,
    work: Path,
    surface: str,
    rtv: bool = False,
    timeout: int = 300,
    backend: str | None = None,
) -> AsanResult:
    """Dispatch S12 to the external checker that matches the execution model."""
    if backend is None:
        backend = backend_for(surface)
    if backend == "cuda":
        return run_sanitizer_gpu(src_mlir, work, surface, rtv=rtv, timeout=timeout)
    return run_asan(src_mlir, work, surface, rtv=rtv, timeout=timeout)


# --------------------------------------------------------------------------
# Assert counting (stage-dependent -- do not inline a grep)
# --------------------------------------------------------------------------
_CF_ASSERT = re.compile(r"\bcf\.assert\b")
_ASSERT_MSG_GLOBAL = re.compile(r"assert_msg_\d+")


def count_asserts(path: Path, lowered: bool) -> int:
    """Count runtime-verification checks in an IR file.

    `lowered=False` -> the file is still at MLIR level, count `cf.assert`.
    `lowered=True`  -> the file is fully lowered to LLVM; `cf.assert` no longer
                       exists and the checks appear as `assert_msg_N` globals.
                       Counting `cf.assert` here would silently return 0.
    """
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return 0
    if lowered:
        return len(set(_ASSERT_MSG_GLOBAL.findall(text)))
    return len(_CF_ASSERT.findall(text))


# RTV bakes the source location into each assert message. In MLIR text the quotes
# arrive octal-escaped -- `Location: loc(\22kernel\22)` -- while the same message
# decoded out of a lowered `assert_msg` global carries plain quotes,
# `Location: loc("kernel")`. Accept both spellings.
_LOC_TAG = re.compile(r'Location:\s*loc\((?:\\22|\\"|")?([A-Za-z_][\w.-]*)')


def count_asserts_by_loc(path: Path) -> dict[str, int]:
    """Attribute each `cf.assert` in RTV-only IR to its source location.

    A composed kernel contains two kinds of code: the operator under audit
    (`loc("kernel")`) and this harness's checksum oracle (`loc("oracle")`). RTV
    instruments *both*, because both lower to `memref.load`/`memref.store`. For
    S9 (remainder = Σ unconditional_guards) only the kernel's guards count -- the
    oracle's are harness artifacts, and folding them in inflates the remainder.
    Measured on relu: 3 kernel guards, 2 oracle guards, so reporting the raw total
    would have overstated S9 by 67% for that category.

    Returns a mapping of location name to guard count.
    """
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return {}
    counts: dict[str, int] = {}
    for line in text.splitlines():
        if "cf.assert" not in line:
            continue
        m = _LOC_TAG.search(line)
        key = m.group(1) if m else "<untagged>"
        counts[key] = counts.get(key, 0) + 1
    return counts


def count_kernel_asserts(path: Path) -> tuple[int, int]:
    """Return ``(kernel_guards, total_guards)`` from RTV-only IR.

    `kernel_guards` is the number to report for S9. Untagged asserts are counted
    as kernel guards, so a missing `loc` degrades to the over-counting behaviour
    rather than silently reporting zero.
    """
    counts = count_asserts_by_loc(path)
    total = sum(counts.values())
    return total - counts.get("oracle", 0), total


def assert_messages(path: Path) -> list[str]:
    """Decode the assert message payloads embedded in lowered IR.

    RTV stores each message in an `assert_msg_N` global as a `dense<"...">`
    tensor of i8. On LLVM 21 the payload is **hex-encoded** (`dense<"0x4552...">`,
    one byte per two hex digits, NUL-terminated); earlier builds used C escapes
    (`\\0A`, `\\22`). Both are handled, because mis-decoding silently yields
    garbage and would corrupt the bounds-vs-alignment distinction that settled the
    M3 x mlir-low question.
    """
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    out = []
    # The declaration is
    #   llvm.mlir.global private constant @assert_msg_3(dense<"0x4552..."> : tensor<189xi8>)
    # so the payload is terminated by `">`, not by `")`. Matching `")` finds
    # nothing and silently returns an empty list, which reads as "RTV emitted no
    # messages" rather than "the decoder is broken".
    for m in re.finditer(r'assert_msg_\d+\(dense<"(.*?)"\s*>', text, re.S):
        raw = m.group(1)
        decoded = _decode_payload(raw)
        if decoded is None:
            out.append(raw)
        else:
            out.append(decoded)
    return out


def _decode_payload(raw: str) -> str | None:
    """Decode one `assert_msg` payload; None if it is not a recognized form."""
    body = raw.replace("\\\n", "").replace("\n", "").strip()
    if body.startswith("0x") or body.startswith("0X"):
        hexstr = body[2:]
        if len(hexstr) % 2:
            return None
        try:
            data = bytes.fromhex(hexstr)
        except ValueError:
            return None
        # Trailing NUL terminator.
        return data.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    try:
        return body.encode("utf-8").decode("unicode_escape")
    except Exception:
        return None


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

# §9.6.1b measured-outcome vocabulary (mutation-specs-v2.md). `outcome` keeps the
# v1 four values; `measured` is the normative five every lane reports. A compile
# failure defaults to `ct-check` (the compiler caught the injected defect) and is
# pinned to `corrupt` only when an audit shows the emitter/base is at fault.
MEASURED_BY_OUTCOME = {
    "n/a": "avoid",
    "compile": "ct-check",
    "runtime": "rt-check",
    "never": "never",
}


@dataclass
class Classification:
    """One mutant's measured behaviour.

    `outcome` is the §6 taxonomy: compile / runtime / never / n/a. Under the
    §9.6.1b measured vocabulary (mutation-specs-v2.md), `runtime` means exactly
    `rt-check`: an *emitted* check the toolchain generated for the bug fired. A
    raw hang, segfault, or abort that is not an emitted check is `never`, not
    `runtime` (rule 3) -- `detected_by` says which.

    `detected_by` splits the v1 `outcome` field, which conflated the mechanism:
    `rt-check` (an emitted check fired) shared `runtime` with a plain hang/crash,
    and `ct-check` (a diagnostic about the bug) shared `compile` with the
    generator-defect `corrupt` state. It is one of:

    * ``rtv-assert``    -- host RTV ``cf.assert`` aborted the run (exit 134)
    * ``gpu-assert``    -- device ``cf.assert`` fired (``CUDA_ERROR_ASSERT``)
    * ``hang``          -- the kernel did not terminate (`EXIT_TIMEOUT`)
    * ``segv``          -- the kernel died on SIGSEGV
    * ``nonzero-exit``  -- any other non-zero exit, no check attributed
    * ``none``          -- it ran to a (possibly wrong) result; nothing fired
    * ``n/a``           -- not expressible, no kernel to run

    `measured` is the §9.6.1b *measured-outcome* vocabulary, normative for every
    lane: `avoid | corrupt | ct-check | rt-check | never`. It is derived from
    `outcome`/`detected_by` via `MEASURED_BY_OUTCOME`, except that an audited
    emitter artifact is pinned to `corrupt` through `measured_override` (rule 1:
    a compile failure is `corrupt` only when it comes from the emitter/base, not
    from the injected defect).

    `manifest` is the §7 ground-truth oracle, renamed `value-changing` / `noop` /
    `undecidable` (§9.6.1b: the old `corrupts` collided with the `corrupt`
    `measured` value).
    """

    outcome: str = "never"
    stage: str = "compile"
    manifest: str = "noop"
    detected_by: str = "none"
    compile_ok: bool = False
    run_ok: bool = False
    ref_check: bool = False
    n_asserts: int = 0
    n_asserts_total: int = 0
    mismatch_count: int = -1
    abort_message: str = ""
    compile_error: str = ""
    notes: str = ""
    measured_override: str = ""

    @property
    def measured(self) -> str:
        """§9.6.1b measured outcome: the vocabulary every lane reports."""
        if self.measured_override:
            return self.measured_override
        if self.detected_by == "emitter-defect":
            return "corrupt"
        return MEASURED_BY_OUTCOME.get(self.outcome, "never")

    def as_record(self) -> dict:
        return asdict(self)


def classify(
    src: Path,
    work: Path,
    surface: str,
    rtv: bool,
    na: bool = False,
    na_reason: str = "",
    run_timeout: int = 60,
    backend: str | None = None,
) -> Classification:
    """Compile, run and classify one kernel.

    The kernel's `@main` returns i32: `0` when the output's checksums (sum and
    sum-of-squares) both match the numpy reference baked in at compose time, `1`
    when either disagrees. That single value is both the ref-check gate and the
    §7 manifest oracle:

    * compile error                      -> outcome=compile,  manifest=value-changing
    * an emitted check fires             -> outcome=runtime,  manifest=value-changing
        - host RTV assert (exit 134):        detected_by=rtv-assert
        - device cf.assert fired:            detected_by=gpu-assert
    * kernel hangs (timeout)             -> outcome=never,    manifest=value-changing
        detected_by=hang
    * kernel dies on SIGSEGV             -> outcome=never,    manifest=value-changing
        detected_by=segv
    * other non-zero exit, no check       -> outcome=never,    manifest=value-changing
        detected_by=nonzero-exit
    * exit 0, mismatch > 0               -> outcome=never,    manifest=value-changing
    * exit 0, mismatch == 0              -> outcome=never,    manifest=noop
                                            (a false success; specs §7.1 discards it)

    `outcome=runtime` therefore means exactly the §9.6.1b `rt-check`. A raw hang
    or crash is `never`, not `runtime` (mutation-specs-v2.md §9.6.1b rule 3): the
    toolchain emitted no check for the bug, so it did not catch it. `detected_by`
    carries the mechanism the v1 `outcome` enum could not.

    Note the `never`/`value-changing` row is the interesting one for this lane: the
    defect survived the verifier *and* RTV, yet the oracle proves the output is
    wrong. That is precisely the silent-bug residue the paper measures.

    `run_timeout` is deliberately short. These kernels are tiny (small-input
    convention, manifest §5.2) and a correct one finishes in milliseconds, so a
    hang is a property of the injected defect rather than of the workload. The M1
    lane needs this more than M2: an out-of-bounds *write* corrupts the heap or
    wraps an index so a loop bound is never reached, and that surfaces as a hang
    instead of a fault.
    """
    if na:
        return Classification(
            outcome="n/a", stage="compile", manifest="noop",
            detected_by="n/a", notes=na_reason
        )

    c = Classification()
    work.mkdir(parents=True, exist_ok=True)
    ll = work / f"{src.stem}.ll.mlir"
    if backend is None:
        backend = backend_for(surface)
    gpu = backend == "cuda"

    # --- compile gate -----------------------------------------------------
    if gpu:
        res = lower_gpu(src, work, surface, rtv, ll)
    else:
        res = run_mlir_opt(pipeline(surface, rtv), src, ll)
    if not res.ok:
        c.compile_ok = False
        c.outcome = "compile"
        c.stage = "compile"
        c.detected_by = "compile-error"
        c.compile_error = (res.stderr or res.stdout).strip()[:2000]
        # A mutant that fails to compile cannot be run, so the §7 oracle is
        # answered statically: the defect is real (it broke the build).
        c.manifest = "value-changing"
        return c
    c.compile_ok = True

    # How many checks did RTV generate? (feeds E3 remainder counting)
    #
    # `n_asserts` counts ONLY guards attributed to loc("kernel"). RTV also
    # instruments this harness's checksum oracle, whose loads/stores are equally
    # real memref accesses; those guards are harness artifacts and must not enter
    # S9. `n_asserts_total` keeps the raw count so the split stays auditable.
    rtv_ir = work / f"{src.stem}.rtv.mlir"
    if rtv:
        rtv_pipe = GPU_RTV_ONLY[(surface, True)] if gpu else RTV_ONLY[surface]
        rres = run_mlir_opt(rtv_pipe, src, rtv_ir)
        if rres.ok:
            c.n_asserts, c.n_asserts_total = count_kernel_asserts(rtv_ir)

    # --- run gate ---------------------------------------------------------
    run = run_kernel(ll, entry_result="i32", timeout=run_timeout, gpu=gpu)
    return _verdict_from_run(run, run_timeout, c)


def _verdict_from_run(
    run: ProcResult, run_timeout: int, c: Classification | None = None
) -> Classification:
    """Map one `mlir-runner` result onto the §6 outcome / §7 manifest taxonomy.

    Factored out of `classify` so `classify_repeat` can apply the *identical*
    verdict logic to each of N runs of an already-lowered kernel. The two must
    not diverge: a verdict rule that lives in only one of them is exactly how the
    lanes drift out of comparability.
    """
    if c is None:
        c = Classification()
    combined = (run.stdout or "") + "\n" + (run.stderr or "")

    # Guard against misreading a JIT symbol error as a detection.
    if run.rc == EXIT_JIT_SYMBOL and any(m in combined for m in _JIT_SYMBOL_MARKERS):
        c.run_ok = False
        c.outcome = "never"
        c.detected_by = "none"
        c.manifest = "noop"
        c.notes = "JIT symbol error (missing shared libs) -- not a detection"
        c.abort_message = combined.strip()[:1000]
        return c

    if run.rc == EXIT_ABORT:
        c.run_ok = False
        c.outcome = "runtime"
        c.stage = "runtime"
        c.detected_by = "rtv-assert"
        c.manifest = "value-changing"
        # First line of the RTV diagnostic, e.g. "^ out-of-bounds access".
        c.abort_message = _first_diagnostic(run.stdout) or combined.strip()[:1000]
        return c

    # GPU RTV: a device `cf.assert` does NOT abort the host process. The CUDA
    # runtime prints the assert (with its baked loc) and a `CUDA_ERROR_ASSERT`
    # from `cuStreamSynchronize`, then `mlir-runner` still exits 0 -- so the exit
    # code alone would misread a caught defect as a clean run. The marker is
    # unique to the CUDA runtime, so this test is inert on the CPU path.
    if "CUDA_ERROR_ASSERT" in combined:
        c.run_ok = False
        c.outcome = "runtime"
        c.stage = "runtime"
        c.detected_by = "gpu-assert"
        c.manifest = "value-changing"
        c.abort_message = _first_gpu_assert(combined) or combined.strip()[:1000]
        return c

    if run.rc == EXIT_TIMEOUT:
        # The kernel never terminated, but no emitted check fired: under
        # mutation-specs-v2.md §9.6.1b rule 3 this is `never`, not `runtime`.
        # `rt-check` is reserved for a check the toolchain *emitted* for the bug;
        # a bare hang is a miss. It stays visible through `detected_by=hang`, and
        # through the M1 lanes' UB-nondeterministic reduction, so a wrapped index
        # is not silently conflated with a clean wrong result.
        c.run_ok = False
        c.outcome = "never"
        c.stage = "runtime"
        c.detected_by = "hang"
        c.manifest = "value-changing"
        c.abort_message = f"TIMEOUT: kernel did not terminate within {run_timeout}s"
        return c

    if run.rc not in (0, EXIT_SEGV):
        # A non-zero exit with no RTV diagnostic and no device assert is a raw
        # crash, not an emitted check -> `never` (§9.6.1b rule 3).
        c.run_ok = False
        c.outcome = "never"
        c.stage = "runtime"
        c.detected_by = "nonzero-exit"
        c.manifest = "value-changing"
        c.abort_message = f"exit {run.rc}: " + combined.strip()[:800]
        return c

    if run.rc == EXIT_SEGV:
        c.run_ok = False
        c.outcome = "never"
        c.stage = "runtime"
        c.detected_by = "segv"
        c.manifest = "value-changing"
        c.abort_message = "SIGSEGV"
        return c

    # --- exit 0: read the mismatch count from stdout ----------------------
    c.run_ok = True
    c.mismatch_count = _parse_return(run.stdout)
    if c.mismatch_count < 0:
        c.notes = "could not parse runner return value"
        c.manifest = "noop"
        return c
    c.ref_check = c.mismatch_count == 0
    c.manifest = "noop" if c.ref_check else "value-changing"
    c.outcome = "never"
    c.stage = "runtime"
    c.detected_by = "none"
    return c


def classify_repeat(
    src: Path,
    work: Path,
    surface: str,
    rtv: bool,
    n: int = 5,
    na: bool = False,
    na_reason: str = "",
    run_timeout: int = 30,
    backend: str | None = None,
) -> list[Classification]:
    """Compile once, run `n` times, return every verdict.

    WHY THIS EXISTS. A dropped-boundary-mask defect (M1.1) injects *undefined
    behavior*: the guardless out-of-bounds write may land idempotently (output
    unchanged -> `noop`), corrupt the heap (`abort`), or wrap an index so a loop
    bound is never reached (`hang`). Which happens depends on heap layout and
    varies run to run. Measured on `mlir-low` small-size: relu/static M1.1 at
    RTV-off gives ~3 `noop` / ~3 `abort` over 6 runs, and transpose/dyn M1.1
    gives ~5 `value-changing` / ~1 `noop`. A single `classify()` therefore *samples the
    UB once* and its verdict -- and any finding derived from it -- is not
    reproducible. This compiles the (deterministic) lowering once and runs it `n`
    times so the caller can aggregate over the distribution.

    Compilation is deterministic, so a compile failure is returned replicated to
    `n` identical verdicts; only the run gate is sampled.
    """
    if na:
        return [
            Classification(outcome="n/a", stage="compile", manifest="noop",
                           detected_by="n/a", notes=na_reason)
        ] * n

    work.mkdir(parents=True, exist_ok=True)
    ll = work / f"{src.stem}.ll.mlir"
    if backend is None:
        backend = backend_for(surface)
    gpu = backend == "cuda"
    if gpu:
        res = lower_gpu(src, work, surface, rtv, ll)
    else:
        res = run_mlir_opt(pipeline(surface, rtv), src, ll)
    if not res.ok:
        c = Classification()
        c.compile_ok = False
        c.outcome = "compile"
        c.stage = "compile"
        c.detected_by = "compile-error"
        c.compile_error = (res.stderr or res.stdout).strip()[:2000]
        c.manifest = "value-changing"
        return [c] * n

    n_asserts = n_asserts_total = 0
    if rtv:
        rtv_ir = work / f"{src.stem}.rtv.mlir"
        rtv_pipe = GPU_RTV_ONLY[(surface, True)] if gpu else RTV_ONLY[surface]
        rres = run_mlir_opt(rtv_pipe, src, rtv_ir)
        if rres.ok:
            n_asserts, n_asserts_total = count_kernel_asserts(rtv_ir)

    out: list[Classification] = []
    for _ in range(n):
        run = run_kernel(ll, entry_result="i32", timeout=run_timeout, gpu=gpu)
        c = _verdict_from_run(run, run_timeout)
        c.compile_ok = True
        c.n_asserts = n_asserts
        c.n_asserts_total = n_asserts_total
        out.append(c)
    return out


def reduce_verdicts(verdicts: list[Classification]) -> tuple[Classification, dict]:
    """Collapse N runs of one (mutant, mode) into one canonical record + census.

    Canonical rule (conservative -- a defect that manifests in *any* run is not a
    false success), applied in order:

    1. every run `compile`            -> `compile`/`value-changing` (deterministic).
    2. any run `runtime` -- an emitted check fired -> `runtime`/`value-changing`.
       The toolchain caught the defect at runtime. NOTE for S1: at RTV-off this
       fault is *incidental UB*, not a generated check -- see mlir-shared/README.md
       "UB-nondeterministic mutants".
    3. else all runs `never`: any `value-changing` -> `never`/`value-changing` (the output
       was wrong, or the run hung/crashed without an emitted check, in at least
       one run); only if *every* run is `noop` -> `never`/`noop`.

    A mutant is therefore a `noop` false success (specs §7.1) only when it is
    noop in *every* run of *every* mode -- the correct reading of "did this
    mutant actually corrupt anything?" under undefined behavior.

    Returns `(canonical, census)` where `census` maps `(outcome, manifest)` to
    its run count, so the caller can report the distribution rather than a
    single sampled verdict.
    """
    census: dict[tuple[str, str], int] = {}
    for v in verdicts:
        census[(v.outcome, v.manifest or "-")] = census.get((v.outcome, v.manifest or "-"), 0) + 1

    if all(v.outcome == "compile" for v in verdicts):
        return verdicts[0], census
    if any(v.outcome == "runtime" for v in verdicts):
        # Pick a representative runtime verdict (prefer one carrying a diagnostic).
        rep = next((v for v in verdicts if v.outcome == "runtime" and v.abort_message), None)
        rep = rep or next(v for v in verdicts if v.outcome == "runtime")
        return rep, census
    # all `never` (or n/a): value-changing wins over noop unless every run is noop.
    if any(v.manifest == "value-changing" for v in verdicts):
        rep = next(v for v in verdicts if v.manifest == "value-changing")
        return rep, census
    return verdicts[0], census


def _first_diagnostic(stdout: str) -> str:
    """Pull the human-readable RTV diagnostic out of runner stdout."""
    for line in (stdout or "").splitlines():
        s = line.strip()
        if s.startswith("^") or "verification failed" in s or "Location:" in s:
            return s
    return ""


def _first_gpu_assert(combined: str) -> str:
    """Pull the device `cf.assert` message out of the CUDA runtime's stderr.

    The runtime prints `... Assertion \`<message>\` failed.` per offending thread,
    where `<message>` is the RTV diagnostic with its baked `loc`.
    """
    for line in (combined or "").splitlines():
        s = line.strip()
        if "Assertion" in s and "failed" in s:
            return s[:1000]
    for line in (combined or "").splitlines():
        if "out-of-bounds" in line or "Location:" in line:
            return line.strip()[:1000]
    return ""


def _parse_return(stdout: str) -> int:
    """`mlir-runner --entry-point-result=i32` prints the value on its own line."""
    for line in reversed((stdout or "").strip().splitlines()):
        s = line.strip()
        if re.fullmatch(r"-?\d+", s):
            return int(s)
    return -1


# --------------------------------------------------------------------------
# Provenance / hashing (non-negotiable cross-cutting fields)
# --------------------------------------------------------------------------


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def short(h: str, n: int = 12) -> str:
    return h[:n]


def settings_hash(category: str, benchmark2: Path) -> str:
    """Hash of the shared settings file for a category (provenance only)."""
    p = benchmark2 / "settings" / f"{category}.md"
    return short(sha1_file(p)) if p.exists() else "MISSING"


# --------------------------------------------------------------------------
# Record emission (schema/record-schema.json)
# --------------------------------------------------------------------------

# There is deliberately no local ENUMS mirror here. An earlier revision kept a
# hand-copied duplicate of the schema's enums, and it silently drifted when the
# coordinator added `paper_category: "hw"` and `stage: "none"` upstream -- the
# mirror then rejected records the real schema accepts. `validate()` reads
# schema/record-schema.json directly via `load_schema()`, which is the only
# source of truth.


def validate(record: dict, kind: str, schema: dict | None = None) -> list[str]:
    """Validate a record against schema/record-schema.json. Returns errors.

    `schema` is accepted and ignored: it is retained only so the existing
    `RecordWriter` call site keeps working. Validating against a caller's own
    parsed copy is how this function came to disagree with the other three
    validators about what `fields` means (see schema/records.py).
    """
    return _records.validate(record, kind)


def load_schema(benchmark2: Path) -> dict:
    return _records.load_schema(benchmark2)


class RecordWriter:
    """Appends JSON-lines to `raw/<stream>.jsonl` (one object per observation)."""

    def __init__(self, raw_dir: Path, stream: str, schema: dict, kind: str):
        self.path = raw_dir / f"{stream}.jsonl"
        self.schema = schema
        self.kind = kind
        self.errors: list[str] = []
        raw_dir.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        errs = validate(record, self.kind, self.schema)
        if errs:
            self.errors.extend(errs)
            print(f"[mlirbench] SCHEMA ERROR: {errs}", file=sys.stderr)
        with self.path.open("a") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def truncate(self) -> None:
        """Clear the stream (idempotent re-runs)."""
        self.path.write_text("")


# --------------------------------------------------------------------------
# Environment self-check
# --------------------------------------------------------------------------


def check_toolchain() -> list[str]:
    """Return a list of problems with the pinned toolchain (empty == healthy).

    The ASan trio is checked separately from the JIT binaries: S1/S9 need only
    `mlir-opt` + `mlir-runner`, whereas S12 additionally needs the native
    lower/instrument/build path. Reporting them apart means a lane missing S12
    tooling says so instead of failing wholesale.
    """
    problems = []
    for b in (MLIR_OPT, MLIR_RUNNER):
        if not b.exists():
            problems.append(f"missing binary: {b}")
    for lib in RUNNER_LIBS.split(","):
        if not Path(lib).exists():
            problems.append(f"missing runner lib: {lib}")
    # The linalg surface defaults to the GPU backend, whose runner needs the
    # CUDA runtime (built only with MLIR_ENABLE_CUDA_RUNNER=ON). Skip when the
    # backend is explicitly pinned to CPU.
    if os.environ.get("MLIR_LINALG_BACKEND", "cuda").lower() != "cpu":
        if not CUDA_RUNTIME_LIB.exists():
            problems.append(
                f"missing CUDA runner lib (MLIR_ENABLE_CUDA_RUNNER=OFF?): {CUDA_RUNTIME_LIB}"
            )
    # Same for the low surface, which is also GPU by default.
    if os.environ.get("MLIR_LOW_BACKEND", "cuda").lower() != "cpu":
        if not CUDA_RUNTIME_LIB.exists():
            problems.append(
                f"missing CUDA runner lib (MLIR_ENABLE_CUDA_RUNNER=OFF?): {CUDA_RUNTIME_LIB}"
            )
        if not COMPUTE_SANITIZER.exists():
            problems.append(
                f"missing compute-sanitizer (see MLIR_COMPUTE_SANITIZER): {COMPUTE_SANITIZER}"
            )
    for b in (MLIR_TRANSLATE, CLANG, LLVM_OPT):
        if not b.exists():
            problems.append(f"missing binary (S12/ASan): {b}")
    if not problems:
        v = _run([str(MLIR_OPT), "--version"])
        m = re.search(r"LLVM version (\S+)", v.stdout)
        if m and m.group(1) != "21.1.0":
            problems.append(
                f"expected LLVM 21.1.0, found {m.group(1)} at {MLIR_OPT}"
            )
    return problems


def describe_toolchain() -> str:
    """One-line human-readable toolchain status for validator output.

    `check_toolchain()` returns a list of problems, which prints as `[]` when
    healthy and tells the reader nothing about what was actually checked. Every
    record carries the toolchain version, so the resolved paths belong in the log
    too -- a number is only reproducible if the binary that produced it is named.
    """
    problems = check_toolchain()
    lin = backend_for("linalg")
    low = backend_for("low")
    any_gpu = "cuda" in (lin, low)
    head = (
        f"{TOOLCHAIN_VERSION}  mlir-opt={MLIR_OPT}"
        f"  linalg-backend={lin}  low-backend={low}"
        + (f" chip={cuda_chip()}" if any_gpu else "")
    )
    if problems:
        return head + "\n  TOOLCHAIN PROBLEMS:\n  - " + "\n  - ".join(problems)
    return head + "  [ok]"
