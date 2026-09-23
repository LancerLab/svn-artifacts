#!/usr/bin/env python3
"""benchmark2/mlir-shared/lane.py — the shared driver for BOTH MLIR lanes.

One implementation, two surfaces. `mlir-linalg` (tensor/linalg entry surface,
M2 shape-contract defects) and `mlir-low` (memref/affine surface, M1
element-access defects) differ only in which emitter, which category list and
which mutation class they measure, so the surface is a parameter rather than a
fork. Everything that could drift between the two lanes — the record shapes, the
census arithmetic, the S1/S8/S9/S12 aggregation — lives here once.

Subcommands (mirroring run.sh.template):

    setup            pin + record the resolved toolchain
    e2               compose and gate every clean kernel the surface expresses
    minimal          E1 mutation battery -> mutants.jsonl (both RTV modes)
    s12              AddressSanitizer supplement -> sanitizer.jsonl
    e3               S8 expressibility + S9 remainder
    collect          raw/*.jsonl -> results/<lane>/*.jsonl (schema-validated)
    stats            results/<lane>/stats.json

E4/E5 are choreo-only (statistics-manifest.md §"Lane ownership") and are not
implemented here.

--------------------------------------------------------------------------
THREE RULES THAT SHAPE EVERY NUMBER BELOW
--------------------------------------------------------------------------

1. **RTV-off and RTV-on are separate records, never merged** (manifest §5.1).
   This is a fairness requirement, not a nicety: RTV is opt-in, and bare MLIR on
   an out-of-bounds `memref.load` silently returns garbage with exit 0. Every
   mutant record therefore carries an `rtv` field, and S1 breaks the detection
   matrix out per mode. The top-level S1 counts use **RTV-on** — a detection
   matrix asks what the toolchain *can* catch, and RTV is part of MLIR — with
   RTV-off exposed alongside it in the same row so the integrator never has to
   guess which was chosen. Picking the higher number as canonical avoids
   understating the SOTA baseline, which would bias the comparison in choreo's
   favour.

2. **S9 counts kernel guards only.** RTV instruments this harness's checksum
   oracle too, because the oracle's loads and stores are real memref accesses,
   but those guards say nothing about the operator under audit. Folding them in
   inflated S9 by ~17% overall and ~67% on relu (measured, `_validate_rtv.py`).
   `count_kernel_asserts()` already excludes `loc("oracle")`; this driver must
   never substitute the total.

3. **The two lanes' numbers are never summed.** mutation-specs.md §7 requires
   the linalg and low S9 remainders be reported separately: the low lane's
   kernels are hand-tiled with an explicit boundary guard, so their guard
   structure is not comparable to the linalg lane's. Same for S1 — there is no
   "MLIR" row, only `mlir-linalg` and `mlir-low`.

--------------------------------------------------------------------------
S12: A REAL SANITIZER MEASUREMENT, NOT `n/a`
--------------------------------------------------------------------------

Each surface measures S12 with the external checker that shares its execution
model, so the `flagged ∧ exercised` number means what it says:

* `low` runs on the GPU, so S12 is `compute-sanitizer --tool memcheck` over the
  same lowered cubin S1 runs. `mlirbench.run_sanitizer_gpu()` lowers with the
  bare (RTV-off) GPU pipeline and counts the device launches actually under
  memcheck; zero launches is a harness failure, not a clean result.
* `linalg` measures S12 with a native AddressSanitizer binary. `run_asan()`
  lowers the bare pipeline to LLVM IR and builds it; that path exists because the
  obvious route is silently broken in four separate ways. See the long comment
  above `run_asan` in mlirbench.py: clang does not instrument `.ll` inputs,
  LLVM's ASan pass skips functions lacking the `sanitize_address` attribute,
  `mlir-translate` emits no target triple (wrong shadow offset, real overflow
  reported as a bogus SEGV), and a native link needs the runner utils the JIT
  resolves dynamically.

Because every one of those failures *looks like success*, both checkers count
their instrumentation/coverage and refuse a "clean" verdict when it is zero. A
clean verdict from an unchecked run is a false negative, which is worse than no
measurement.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent                  # mlir-shared/
ROOT = HERE.parent                                      # benchmark2/
sys.path.insert(0, str(HERE))

import compose as C
import emit as E
import emit_low as L
import mlirbench as B
import mutate as M

# --------------------------------------------------------------------------
# The mutation-class axis is NOT restated here.
# --------------------------------------------------------------------------
# This file used to iterate a literal ("M1", "M2", "M3") in three places, left
# over from the pre-revision spec. The effect was that M4 was absent from these
# lanes' detection matrices WITHOUT ANYTHING SAYING SO -- and an absent class is
# not the same claim as an `n/a` class. `n/a` means the lane was run and its
# surface cannot express the defect; absence may mean exactly that, or it may
# mean nobody looked. schema/class-axis.json is the one definition of which it
# is, `AX.lane_status("mlir-linalg")` gives this lane's verdict per class, and
# schema/check_class_axis.py fails the build if this module drifts from it.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from schema import class_axis as AX                             # noqa: E402
from schema import method_taxonomy as T                         # noqa: E402

# Helpers that resolve this file's SHARED surface table against the axis, so
# the two surfaces (`linalg` -> lane `mlir-linalg`, `low` -> lane `mlir-low`)
# cannot disagree with the axis about which class is n/a and which is
# uncompared.
MUTATION_CLASSES = AX.mutation_classes()


def _na_classes(lane: str) -> tuple[str, ...]:
    """Classes this lane RAN and cannot express. Each gets an n/a row."""
    return tuple(AX.lane_where(lane, "n/a"))


def _uncompared_classes(lane: str) -> tuple[str, ...]:
    """Classes this lane was never run against. Each gets NO row, and is
    declared in `S1_declared_uncompared` instead."""
    return tuple(AX.lane_where(lane, "uncompared"))


def _attr_class(rec: dict) -> str:
    """The class a mutant record belongs to, resolved from its spec_id.

    NOT `rec["class"]`: when a spec is re-homed to another class (M1.6 ->
    M4-d), the writer keeps stamping the old class while the taxonomy files the
    family under the new one. Reading the stamp files those injections under M1
    and leaves the M4 cell empty even though its instances were generated. The
    taxonomy is the one authority (`T.family_of`), so resolve through it and only
    fall back to the stamp when the spec_id names no family (an avoided or
    attribution_only operator such as M2.12).
    """
    spec = rec.get("spec_id")
    if not spec:
        # The sanitizer stream carries no `spec_id`; its `mutant_id` ends in
        # `...-<spec_id>-<rtv>` (e.g. `mlir-low-relu-static-M1.6-off`), so the
        # spec is recoverable without re-running the lane. Parsing the id is the
        # fallback, not the rule: prefer an explicit spec_id when there is one.
        m = re.search(r"-(M\d+\.\d+)(?:-|$)", rec.get("mutant_id", ""))
        spec = m.group(1) if m else ""
    fam = T.family_of(spec)
    return T._FAM[fam]["class"] if fam else rec.get("class", "")

# --------------------------------------------------------------------------
# Surface table — the ONLY place the two lanes differ
# --------------------------------------------------------------------------

# mutation-specs.md §4's translation table, made executable:
#   M1 OOB   -> linalg n/a (no memory at the tensor level); low affine/memref
#   M2 shape -> linalg rank/shape mismatch (its primary target); low n/a
#   M3 hw    -> n/a on both (measured §4.1; was "partial" before measurement)
SURFACES = {
    "linalg": {
        "toolchain": "mlir-linalg",
        "surface": "linalg",
        "lane": "mlir-linalg",
        "emitter": E.emit_kernel,
        "categories": ["matmul", "relu", "softmax", "transpose", "concat",
                       "layer_normalization", "elemwise_add",
                       "transpose_square", "pad", "reshape", "broadcast",
                       # Rank/axis variants that host the M2-e second surface and
                       # the M2-d/g/h families' budget. They compose and gate
                       # exactly like their base categories; only the battery
                       # treats them as the mutation-only hosts they are.
                       "transpose_cube",
                       "pad_last", "pad_mid", "pad_r3",
                       "reshape_r3", "reshape_r4", "reshape_r5",
                       "broadcast_r2", "broadcast_r4", "broadcast_r5"],
        # The mutation battery is NOT the composed set. `categories` above is
        # what E2 gates and what S8/S9 report over; `battery_cats` is
        # `compose.M2_CATS` (the level-1/level-2 M2 categories). relu and
        # transpose compose and carry RTV guards but are not M2 injection
        # targets; iterating `categories` here silently produced 70 injections
        # instead of the spec-count arithmetic.
        "battery_cats": list(C.M2_CATS),
        # The historical §5.1 Cartesian arithmetic (10 spec ids x cats x 2
        # shapes), kept only for reference. M2 no longer compares against it:
        # `select_m2` now holds each FAMILY at N_PER_FAMILY, and the census
        # checks the per-family budget instead (see below), so the measured
        # count legitimately sits below this Cartesian product.
        "expected_injected": {"1": 54, "2": 90},
        "klass": "M2",
        # S12 tool: the linalg surface lowers to LLVM and builds a native ASan
        # binary (see the S12 note at the top of this file). It is not switched to
        # compute-sanitizer with the S1 backend because its S12 measurement is
        # already committed and is a separate question from where the kernels run.
        "sanitizer": "asan",
        "specs": C.M2_SPECS,
        "spec_ids": C.M2_SPEC_IDS,
        "level_of": C.LEVEL_OF,
        # What this surface cannot express at all, and WHY -- is not recorded
        # here. The reason prose lives once, in schema/class-axis.json's
        # `na_reasons[lane][class]`, and is read back with AX.na_reason(). It
        # used to be a dict in this file, hand-worded per surface, and the two
        # surfaces had drifted into different wording for the same M3 reason.
        # The class SET is AX.lane_where(lane, "n/a") for the same reason.
        #
        # M4 is absent by design and is NOT an n/a: this lane was never run
        # against it, which is `uncompared` (see `_uncompared_classes`), and it
        # gets no row at all rather than a fabricated 0.
    },
    "low": {
        "toolchain": "mlir-low",
        "surface": "low",
        "lane": "mlir-low",
        "emitter": L.emit_kernel_low,
        "categories": list(L.LOW_CATS),
        # `categories` is the composed set E2/S8/S9 gate; `battery_cats` is what
        # the M1 battery and S12 walk. They coincide for the four operator
        # categories, and `battery_cats` additionally carries the mutation-only
        # carrier hosts of family M1-f (spec M1.19), which have no clean-composed
        # contract beyond the narrow-carrier mutant they exist to host. See
        # `m1_cell` for the pairing rule.
        "battery_cats": list(L.LOW_CATS) + list(C.M1_CARRIER_CATS),
        # Pinned, NOT derived from the specs being iterated. v1 §5.1 was 6 specs
        # x 4 cats x 2 shapes = 48; the v2.1 additions the memref surface can
        # realise (M1.11 family M1-h, M1.12 family M1-d, M1.14 family M1-e) add
        # 3 specs x 4 x 2 = 24. Family M1-a is then trimmed from its 3 realising
        # specs (M1.1/2/3, 24) to one (M1.1, 8) so it matches the single-spec
        # budget every other family holds, and M1.20 (family M1-g) and M1.19
        # (family M1-f) are added as one spec each. `m1_cell` pairs M1.19 only
        # with the four carrier hosts and every other spec only with the four
        # operator categories, so the battery is 8 specs x 4 cats x 2 shapes
        # (operator categories, 64) + 1 spec x 4 carrier hosts x 2 shapes (8)
        # = 72. Every M1 record is level-1.
        "expected_injected": {"1": 72},
        "klass": "M1",
        # The kernel now executes on the device, so S12 must use the checker that
        # shares that execution model: `compute-sanitizer --tool memcheck` over
        # the lowered cubin. ASan (the `linalg` surface's tool) instruments a
        # natively-linked host binary and cannot observe a device access at all,
        # so it would report every mutant clean -- a false negative, not a
        # measurement.
        "sanitizer": "compute-sanitizer",
        "specs": C.M1_LOW_SPECS,
        "spec_ids": [m.spec_id for m in C.M1_LOW_SPECS],
        # mutation-specs.md §5's M1 minimal set is exactly the four categories
        # this lane composes, so every M1 record is level-1. The §5 level-2 M1
        # additions (max_pool2d, conv2d, embedding, batch_norm) have no
        # low-level emitter yet — see the breadth note in stats.json.
        "level_of": {c: "1" for c in list(L.LOW_CATS) + list(C.M1_CARRIER_CATS)},
        # n/a reasons come from the axis here too; see the note on the linalg
        # surface above.
    },
}

# Every class the axis holds at `n/a` must have a reason recorded in the axis,
# and every class it holds at `uncompared` must have none. Asserted at import
# so a lane cannot silently grow a class it never ran.
for _key, _cfg in SURFACES.items():
    _lane = _cfg["lane"]
    for _cls in _na_classes(_lane):
        if not AX.na_reason(_lane, _cls).strip():
            raise AssertionError(
                f"surface {_key!r}: class {_cls} is n/a but "
                f"schema/class-axis.json records no reason for it")
    # NOTE: this file used to assert that every surface had >= 1 `uncompared`
    # class, to prove the declaration path below is exercised. As of 2026-09-23
    # every class is measured on both surfaces (M4 was the last `uncompared`),
    # so that assertion would fire on a correct axis. The handling stays -- a
    # future class may be declared `uncompared` again -- it is simply no longer
    # required to be non-empty. `check_class_axis.py` still reconciles whatever
    # the axis declares against each lane's stats.
    fwd = {c["id"]: c["obligation"] for c in AX.axis()["classes"]}
    if not set(MUTATION_CLASSES) <= set(fwd):
        raise AssertionError("the class axis is missing an obligation mapping")

# manifest §1 counts 15 categories. These lanes compose 7 (linalg) and 4 (low).
# The user's ruling: ship green at the composed subset and document the breadth
# gap explicitly rather than block on it. The missing eight are named in
# stats.json so the coordinator sees a flagged follow-up, not a silent hole.
ALL_CATEGORIES = ["batch_norm", "concat", "conv2d", "elemwise_add", "embedding",
                  "gelu", "layer_normalization", "matmul", "max_pool2d",
                  "reduce_mean", "relu", "reshape", "sigmoid", "softmax",
                  "transpose"]

# M1.1's dropped-boundary-mask defect injects undefined behavior, so a single
# run samples one of {noop, abort, hang} non-reproducibly. N=16 is sized against
# the one cell whose recorded outcome can realistically flip: relu/static M1.1 at
# RTV-off, p(noop)=0.532 pooled over 94 runs across five independent draws, so
# the never/noop tally cell is wrong with probability ~4.1e-5 (95% interval
# ~1.5e-6..6.1e-4 — p is itself only known by sampling, so quote the interval,
# not a point estimate). Three other cells (transpose/static + transpose/dynamic
# M1.1, layer_norm/dynamic M1.2, all RTV-off) have also produced a mixed
# distribution at least once, but at p(noop)<=1/16 their flip probability is
# <=1e-19: nondeterministic in the strict sense, stable in practice. The linalg
# M2 battery has no such cell — its 128 records are all compile/never with zero
# runtime outcomes, so nothing samples UB and it runs once. Both counts are
# env-overridable for a quick check or a deeper pass.
REPEAT = {"low": int(os.environ.get("M1_REPEAT", "16")),
          "linalg": int(os.environ.get("M2_REPEAT", "1"))}
RUN_TIMEOUT = int(os.environ.get("MLIR_RUN_TIMEOUT", "15"))


def m1_cell(cat: str, spec_id: str) -> bool:
    """Whether `spec_id` may be injected into `cat` on the M1 low battery.

    Family M1-f (spec M1.19) is the only spec hosted by the mutation-only
    carrier categories (`compose.M1_CARRIER_CATS`), and it is hosted by no
    operator category; every other M1 spec is hosted by the operator categories
    and not by the carriers. Pairing them the other way is either inert (a narrow
    index over an axis the carrier width already spans) or would misattribute the
    defect to an operator with no clean carrier form. `battery_cats` therefore
    spans more than `categories`, and this predicate is the pairing rule.
    """
    return (spec_id == "M1.19") == (cat in C.M1_CARRIER_CATS)


def select_m2(cats: list[str], specs, size: str = "small") -> set[tuple]:
    """Choreo-style per-family selection for the M2 battery.

    Mirrors `choreo/gen_mutants.py:select`: `N = N_PER_FAMILY` (8) decomposes as
    `N_KERNELS x N_REALISATIONS`, so a family realises each declared spec_id at
    least once and then fills round-robin to `N`, capping every
    `(family, category)` at `N_REALISATIONS`. A candidate the surface cannot
    express (`NotApplicable`) or that changes nothing (`AssertionError`) is not a
    candidate at all, so an n/a cell never consumes a slot. Returns the selected
    `(category, shape, spec_id)` injection keys.
    """
    n_fam = T.N_PER_FAMILY
    n_real = T.N_REALISATIONS
    by_spec: dict[str, list] = defaultdict(list)
    for mut in specs:
        by_spec[mut.spec_id].append(mut)

    # Candidate injections, with expressibility probed on the clean case.
    cands: list[tuple[str, str, str, str]] = []  # family, spec_id, cat, shape
    for cat in cats:
        for dynamic in (False, True):
            shape = "dynamic" if dynamic else "static"
            clean = C.make_case(cat, size=size, dynamic=dynamic)
            for sid, muts in by_spec.items():
                for mut in muts:
                    try:
                        M.apply(clean, mut)
                        cands.append((T.family_of(sid), sid, cat, shape))
                        break
                    except (M.NotApplicable, AssertionError):
                        continue

    by_fam: dict[str, list] = defaultdict(list)
    for c in cands:
        by_fam[c[0]].append(c)

    selected: set[tuple] = set()
    for fam, cc in by_fam.items():
        cap: Counter = Counter()
        chosen: list[tuple] = []

        def pick(pred) -> None:
            for c in sorted(cc, key=lambda z: (cap[z[2]], z[2], z[3], z[1])):
                if c in chosen or cap[c[2]] >= n_real or not pred(c):
                    continue
                chosen.append(c)
                cap[c[2]] += 1
                return

        # Stage 1 -- every declared spec_id in the family realised at least once.
        for sid in by_spec:
            if T.family_of(sid) != fam:
                continue
            pick(lambda c, sid=sid: c[1] == sid)
        # Stage 2 -- fill round-robin to N, capped per category.
        while len(chosen) < n_fam:
            before = len(chosen)
            for c in sorted(cc, key=lambda z: (cap[z[2]], z[2], z[3], z[1])):
                if len(chosen) >= n_fam:
                    break
                if c in chosen or cap[c[2]] >= n_real:
                    continue
                chosen.append(c)
                cap[c[2]] += 1
            if len(chosen) == before:
                break
        selected.update((c[2], c[3], c[1]) for c in chosen)
    return selected


def _one_variant_per_injection(specs, sel: set | None) -> dict[tuple, object]:
    """Pick a single mutant per selected injection when a spec_id has variants.

    M2.5 is declared twice ("partial-write", "duplicate-write") and both mutants
    share the injection identity `M2.5`, so `select_m2` sees one candidate per
    host and fills a whole family's worth of hosts. Emitting every variant on
    every host (the earlier behaviour) then produced TWO programs per injection,
    giving family M2-f 16 instances and 32 records -- twice the N=8 budget its
    seven sibling families meet. The budget is `N_KERNELS x N_REALISATIONS` over
    hosts, not variants, so exactly one program is emitted per injection and the
    variants are dealt round-robin across the family's sorted injection keys;
    both variants still appear, just not twice on the same host.
    """
    if sel is None:
        return {}
    by_sid: dict[str, list] = defaultdict(list)
    for mut in specs:
        by_sid[mut.spec_id].append(mut)
    chosen: dict[tuple, object] = {}
    for sid, muts in by_sid.items():
        if len(muts) < 2:
            continue
        for i, k in enumerate(sorted(k for k in sel if k[2] == sid)):
            chosen[k] = muts[i % len(muts)]
    return chosen


# The MLIR diagnostic is the interesting part of a compile error; the leading
# `path:line:col:` prefix is noise in a summary table.
_LOC = re.compile(r"^.*?\.mlir:\d+:\d+:\s*", re.MULTILINE)

# S1's outcome strength order. An injection is counted at its MOST-detected
# variant (a spec with variants deals them across hosts, so an ordinary
# injection carries one outcome; the reduction still applies if a future spec
# emits several). Strength runs compile (the verifier rejected it outright)
# > runtime (a generated check fired) > never (it ran and silently produced
# wrong output).
_STRENGTH = {"compile": 3, "runtime": 2, "never": 1, "n/a": 0}


def log(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


def _detail(c: B.Classification) -> str:
    raw = c.abort_message or c.compile_error or c.notes or ""
    return " ".join(_LOC.sub("", raw).split())[:120]


class Lane:
    """One surface's driver. Holds the resolved paths and the toolchain stamp."""

    def __init__(self, surface_key: str):
        if surface_key not in SURFACES:
            raise SystemExit(f"unknown surface {surface_key!r}; "
                             f"expected one of {sorted(SURFACES)}")
        self.key = surface_key
        self.cfg = SURFACES[surface_key]
        self.toolchain = self.cfg["toolchain"]
        self.surface = self.cfg["surface"]
        self.lane_dir = ROOT / self.toolchain
        self.raw = self.lane_dir / "raw"
        self.results = ROOT / "results" / self.toolchain
        self.schema = B.load_schema(ROOT)
        self.version = B.TOOLCHAIN_VERSION
        # Where the kernels actually execute for this surface. `low` is a CUDA
        # lane: it has no CPU fallback in the committed measurement.
        self.backend = B.backend_for(self.surface)
        self.gpu = self.backend == "cuda"
        # The composed categories own a settings file; a mutation-only carrier
        # host borrows its base operator's settings hash (`x_carrier` -> `x`), so
        # the carrier records carry a real provenance hash rather than MISSING.
        # Union of both sets: `battery_cats` is a superset of `categories` on the
        # low surface and a subset on some tensor surfaces.
        self.settings = {
            c: B.settings_hash(C.base_category(c), ROOT)
            for c in set(self.cfg["categories"]) | set(self.cfg["battery_cats"])
        }

    # -- record helpers ----------------------------------------------------

    def _base(self, category: str) -> dict:
        """The non-negotiable cross-cutting provenance fields (manifest §8)."""
        return {"toolchain": self.toolchain,
                "toolchain_version": self.version,
                "settings_hash": self.settings.get(category, "MISSING")}

    def _writer(self, stream: str, kind: str, truncate: bool = True) -> B.RecordWriter:
        w = B.RecordWriter(self.raw, stream, self.schema, kind)
        if truncate:
            w.truncate()
        return w

    # -- setup -------------------------------------------------------------

    def cmd_setup(self) -> int:
        problems = B.check_toolchain()
        self.raw.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)
        stamp = {
            "toolchain": self.toolchain,
            "surface": self.surface,
            "llvm_version": self.version,
            "llvm_bin": str(B.LLVM_BIN),
            "llvm_lib": str(B.LLVM_LIB),
            "mlir_opt": str(B.MLIR_OPT),
            "mlir_runner": str(B.MLIR_RUNNER),
            "mlir_translate": str(B.MLIR_TRANSLATE),
            "clang": str(B.CLANG),
            "opt": str(B.LLVM_OPT),
            "runner_libs": B.RUNNER_LIBS,
            "asan_target": B.target_lines(),
            "device": (f"{self.backend} " +
                       (f"({B.cuda_chip()}; JIT of the lowered cubin on the "
                        "device)" if self.gpu else
                        "(JIT on host; no GPU dependency)")),
            "categories_composed": self.cfg["categories"],
            "categories_missing": sorted(set(ALL_CATEGORIES)
                                         - set(self.cfg["categories"])),
            "problems": problems,
        }
        (self.raw / "setup.json").write_text(json.dumps(stamp, indent=2) + "\n")
        log(B.describe_toolchain())
        if problems:
            log(f"[{self.toolchain}] setup: {len(problems)} TOOLCHAIN PROBLEM(S)")
            for p in problems:
                log(f"  !! {p}")
            return 1
        log(f"[{self.toolchain}] setup ok — {len(self.cfg['categories'])} "
            f"categories composed, {len(stamp['categories_missing'])} missing "
            f"(documented breadth gap)")
        return 0

    # -- E2: compose + gate ------------------------------------------------

    def _emit_clean(self, cat: str, dynamic: bool, size: str) -> str:
        case = C.make_case(cat, size=size, dynamic=dynamic)
        return self.cfg["emitter"](case)

    def _emit_mutant(self, mcase, clean, structural) -> str:
        """Emit a mutant against the CLEAN case's reference.

        `ref_case` must always be the unmutated case: comparing a mutant against
        its own mutated reference would make every mutant look correct and turn
        the whole §7 ground-truth oracle into a tautology. Both emitters take it.
        """
        return self.cfg["emitter"](mcase, ref_case=clean,
                                   structural=structural)

    def cmd_e2(self, size: str = "small") -> int:
        """Compose and gate every clean kernel this surface expresses.

        The gate is the bare (RTV-off) pipeline: does it lower, does it run, does
        the output match the numpy reference baked in at compose time. The
        RTV-on variant is measured too and recorded as an extra field, because
        §5.1 requires both modes be reported — but one record per
        (category, shape, size) keeps `totals.kernels` a meaningful census of the
        composed suite rather than double-counting it.

        Both sizes are committed, matching the `iree` lane (311 small + 311 full
        in its `kernels.jsonl`). The writer truncates, so the other size's
        records are carried across explicitly — running `--full` after `--small`
        must not silently delete the small-size gate. Re-running the same size is
        idempotent: its old records are dropped and replaced.

        Records are written in a canonical (size, category, shape) order rather
        than "kept rows then fresh rows", so that re-running the documented
        `./run.sh all` path reproduces `kernels.jsonl` byte-for-byte. Without the
        sort, a second run emits full-then-small where the first emitted
        small-then-full: identical content, different order, and a spurious diff
        on a committed artifact.
        """
        kept = [r for r in self._load("kernels.jsonl") if r.get("size") != size]
        fresh: list[dict] = []
        n_bad = 0
        for cat in self.cfg["categories"]:
            for dynamic in (False, True):
                shape = "dynamic" if dynamic else "static"
                try:
                    src = self._emit_clean(cat, dynamic, size)
                except Exception as exc:                      # noqa: BLE001
                    log(f"  {cat:22s} {shape:8s} EMIT-FAIL "
                        f"{type(exc).__name__}: {exc}")
                    n_bad += 1
                    continue

                rec = self._base(cat)
                rec.update({
                    "category": cat,
                    "kernel": shape,
                    "shape": shape,
                    "size": size,
                    "gpu_device": self.backend,
                    "exclusive": "false",
                    "kernel_hash": B.short(B.sha1_text(src)),
                })

                with tempfile.TemporaryDirectory() as td:
                    tmp = Path(td)
                    f = tmp / f"{cat}_{shape}.mlir"
                    f.write_text(src)
                    bare = B.classify(f, tmp, self.surface, False,
                                      run_timeout=RUN_TIMEOUT)
                    checked = B.classify(f, tmp, self.surface, True,
                                         run_timeout=RUN_TIMEOUT)

                rec["compile"] = "ok" if bare.compile_ok else "fail"
                rec["run"] = "ok" if bare.run_ok else "crash"
                rec["ref_check"] = "pass" if bare.ref_check else "fail"
                rec["rtv_on_run"] = "ok" if checked.run_ok else "crash"
                rec["rtv_on_ref_check"] = "pass" if checked.ref_check else "fail"
                rec["rtv_on_kernel_guards"] = checked.n_asserts
                if not bare.ref_check:
                    rec["detail"] = _detail(bare) or "ref-check failed"
                fresh.append(rec)

                ok = rec["compile"] == "ok" and rec["ref_check"] == "pass"
                if not ok:
                    n_bad += 1
                log(f"  {cat:22s} {shape:8s} {size:5s} "
                    f"compile={rec['compile']:4s} run={rec['run']:5s} "
                    f"ref={rec['ref_check']:4s} "
                    f"rtv_on={rec['rtv_on_ref_check']:4s} "
                    f"guards={rec['rtv_on_kernel_guards']:3d} "
                    f"{'' if ok else '<-- GATE FAIL'}")

        # canonical order: small before full, then the category order, then
        # static before dynamic. Records lacking a size sort last so a legacy
        # file cannot be silently reordered ahead of the current run.
        _SIZE_ORDER = {"small": 0, "full": 1}
        _CAT_ORDER = {c: i for i, c in enumerate(self.cfg["categories"])}

        def _key(r: dict) -> tuple:
            return (_SIZE_ORDER.get(r.get("size"), 9),
                    _CAT_ORDER.get(r.get("category"), 99),
                    r.get("category", ""),
                    0 if r.get("shape") == "static" else 1)

        w = self._writer("kernels", "kernel")
        for r in sorted(kept + fresh, key=_key):
            w.write(r)

        if w.errors:
            log(f"[{self.toolchain}] e2: {len(w.errors)} SCHEMA ERROR(S)")
            for e in w.errors[:10]:
                log(f"  !! {e}")
            return 1
        log(f"[{self.toolchain}] e2 [{size}]: {n_bad} gate failure(s)")
        return 1 if n_bad else 0

    # -- E1: mutation battery ---------------------------------------------

    def cmd_minimal(self, level2: bool = False, size: str = "small") -> int:
        """Emit one mutant record per (category, shape, mutant, RTV mode).

        TWO DIFFERENT COUNTS, deliberately kept apart (mutation-specs.md §0/§2):

        * an **injection** is one (category, shape, spec) triple — this is what
          S1's `n_injected` reports, and M2's spec 5 counts once even though it
          declares two manifestations (dealt one per host, so the two variants
          never stack);
        * a **record** is one (category, shape, mutant, rtv-mode) tuple.

        §5.1's enumeration arithmetic is owner-approved and must NOT be trimmed
        to hit N=40 exactly: linalg M2 = 5 specs x 5 cats x 2 shapes = 50
        injections -> 54 mutants -> 120 records; low M1 = 8 operator specs x 4
        cats x 2 shapes (64) + spec M1.19 x 4 carrier hosts x 2 shapes (8) = 72
        injections -> 72 mutants -> 144 records (the M1.19 pairing is `m1_cell`).
        The +10 / +24 overshoot is recorded explicitly in stats.json. (v1 shipped
        6 M1 specs; the v2.1 additions M1.11/M1.12/M1.14 raise it to 9, and
        M1.2/M1.3 are then dropped so family M1-a holds one spec like every other
        family; M1.20 realises M1-g and M1.19 realises M1-f.)
        """
        w = self._writer("mutants", "mutant")
        klass = self.cfg["klass"]
        cats = list(self.cfg["battery_cats"])
        if klass == "M2" and not level2:
            cats = list(C.LEVEL1_M2_CATS)

        # Per-family selection (choreo/gen_mutants.py:select), so a family holds
        # N_PER_FAMILY instances rather than the whole spec x category x shape
        # Cartesian. Without it M2-a alone holds 40 -- not comparable to choreo's
        # 8-per-family cells. Only M2 is selected; the M1 low battery is already
        # at its §5.1 contract.
        sel = (select_m2(cats, self.cfg["specs"], size=size)
               if klass == "M2" else None)
        variant = _one_variant_per_injection(self.cfg["specs"], sel)

        n_repeat = REPEAT[self.key]
        # injection key -> {rtv: [outcomes of that injection's variants]}
        inj: dict[tuple[str, str, str], dict[bool, list[str]]] = defaultdict(
            lambda: {False: [], True: []})
        problems: list[str] = []
        n_records = 0

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for cat in cats:
                lv = self.cfg["level_of"].get(cat, "1")
                for dynamic in (False, True):
                    shape = "dynamic" if dynamic else "static"
                    clean = C.make_case(cat, size=size, dynamic=dynamic)
                    clean_src = self._emit_mutant(clean, clean, None)
                    for mut in self.cfg["specs"]:
                        if klass == "M1" and not m1_cell(cat, mut.spec_id):
                            continue
                        if sel is not None and (cat, shape, mut.spec_id) not in sel:
                            continue
                        if mut is not variant.get((cat, shape, mut.spec_id), mut):
                            continue
                        ikey = (cat, shape, mut.spec_id)
                        try:
                            mcase, structural = M.apply(clean, mut)
                            # Emit inside the same try: NotExpressible is raised
                            # by the EMITTER, not by apply(). The Case
                            # perturbation is well-formed, but this surface
                            # cannot realize the defect on this category's rank
                            # (e.g. transposed-stride needs two axes to swap).
                            src = self._emit_mutant(mcase, clean, structural)
                            if src == clean_src:
                                raise AssertionError(
                                    "emitter produced a module byte-identical "
                                    "to the clean kernel -- the perturbation "
                                    "is invisible to this surface")
                        except M.NotApplicable as exc:
                            # §6: n/a cells are counted and never re-balanced.
                            for rtv in (False, True):
                                rec = self._mutant_rec(
                                    cat, lv, shape, mut, rtv,
                                    B.Classification(outcome="n/a", stage="none",
                                                     manifest="noop",
                                                     notes=str(exc)))
                                w.write(rec)
                                n_records += 1
                            inj[ikey][False].append("n/a")
                            inj[ikey][True].append("n/a")
                            log(f"  {cat:22s} {shape:8s} {mut.mutant_id:16s} "
                                f"n/a  ({exc})")
                            continue
                        except AssertionError as exc:
                            # A mutation that changes nothing would be recorded as
                            # a `noop` false success and silently shrink N
                            # (specs §7.1). That is a harness bug, not a result.
                            problems.append(f"NOOP mutation {mut.mutant_id} on "
                                            f"{cat}/{shape}: {exc}")
                            continue
                        except L.NotExpressible as exc:
                            for rtv in (False, True):
                                rec = self._mutant_rec(
                                    cat, lv, shape, mut, rtv,
                                    B.Classification(outcome="n/a", stage="none",
                                                     manifest="noop",
                                                     notes=str(exc)))
                                w.write(rec)
                                n_records += 1
                            inj[ikey][False].append("n/a")
                            inj[ikey][True].append("n/a")
                            log(f"  {cat:22s} {shape:8s} {mut.mutant_id:16s} "
                                f"n/a  (not expressible: {exc})")
                            continue

                        # Emitted ONCE per mutant; only classification depends on
                        # the RTV mode.
                        f = tmp / f"{cat}_{shape}_{mut.mutant_id}.mlir"
                        f.write_text(src)
                        khash = B.short(B.sha1_text(src))

                        for rtv in (False, True):
                            verdicts = B.classify_repeat(
                                f, tmp, self.surface, rtv, n=n_repeat,
                                run_timeout=RUN_TIMEOUT)
                            c, census = B.reduce_verdicts(verdicts)
                            rec = self._mutant_rec(cat, lv, shape, mut, rtv, c)
                            rec["kernel_hash"] = khash
                            if n_repeat > 1:
                                rec["n_repeat"] = str(n_repeat)
                                rec["distribution"] = ", ".join(
                                    f"{n}x {o}/{m}" for (o, m), n
                                    in sorted(census.items()))
                            w.write(rec)
                            n_records += 1
                            inj[ikey][rtv].append(c.outcome)
                            log(f"  {cat:22s} {shape:8s} {mut.mutant_id:16s} "
                                f"rtv={'on ' if rtv else 'off'} "
                                f"{c.outcome:8s} {c.manifest or '-':9s} "
                                f"{_detail(c)}")

        # ---- injection census (spec level; feeds S1 n_injected) -----------
        census_rows = {}
        for rtv in (False, True):
            tally = Counter()
            for outs in inj.values():
                tally[_reduce_injection(outs[rtv])] += 1
            census_rows[rtv] = tally

        n_inj = len(inj)
        lvl = "2" if level2 else "1"
        if sel is None:
            # M1 (low): still the PINNED §5.1 contract number, never a count
            # derived from `cats` — that would be self-referential and would pass
            # no matter which categories were iterated.
            expected = self.cfg["expected_injected"][lvl]
            log(f"[{self.toolchain}] minimal: {n_records} records over {n_inj} "
                f"injections ({klass}); §5.1 expects {expected} "
                f"{'OK' if n_inj == expected else '<-- MISMATCH'}")
            if n_inj != expected:
                problems.append(
                    f"injection census {n_inj} != §5.1 contract {expected} "
                    f"(level {lvl}, {len(cats)} categories x "
                    f"{len(self.cfg['spec_ids'])} specs x 2 shapes)")
        else:
            # M2: the contract is per FAMILY, not per battery -- each family
            # holds at most N_PER_FAMILY injections, and reaches it whenever its
            # expressible supply allows. `expected` is the measured total, so it
            # is recorded for the arithmetic but is not the pass condition.
            fam_count = Counter(T.family_of(k[2]) for k in inj)
            expected = n_inj
            over = {f: n for f, n in fam_count.items() if n > T.N_PER_FAMILY}
            log(f"[{self.toolchain}] minimal: {n_records} records over {n_inj} "
                f"injections ({klass}); N={T.N_PER_FAMILY} per family over "
                f"{len(fam_count)} families "
                f"{'OK' if not over else '<-- OVER BUDGET'}")
            log("    per family: " + ", ".join(
                f"{f}={n}" for f, n in sorted(fam_count.items())))
            if over:
                problems.append(
                    f"family budget exceeded: {over} (N={T.N_PER_FAMILY})")
        for rtv in (False, True):
            log(f"    rtv={'on ' if rtv else 'off'}: "
                f"{dict(sorted(census_rows[rtv].items()))}")

        # Persist the census so `stats` does not have to re-derive it and so the
        # arithmetic is auditable from committed raw JSON (manifest §8).
        (self.raw / "minimal-census.json").write_text(json.dumps({
            "class": klass,
            "n_records": n_records,
            "n_injected": n_inj,
            "expected_injected": expected,
            "n_target_per_class": C.N_TARGET_PER_CLASS,
            "delta_vs_target": n_inj - C.N_TARGET_PER_CLASS,
            "categories": cats,
            "level2": level2,
            "size": size,
            "n_repeat": n_repeat,
            "by_rtv": {("on" if r else "off"): dict(sorted(v.items()))
                       for r, v in census_rows.items()},
            "problems": problems,
        }, indent=2) + "\n")

        if w.errors:
            log(f"[{self.toolchain}] minimal: {len(w.errors)} SCHEMA ERROR(S)")
            for e in w.errors[:10]:
                log(f"  !! {e}")
            return 1
        if problems:
            log(f"[{self.toolchain}] minimal: {len(problems)} PROBLEM(S)")
            for p in problems:
                log(f"  !! {p}")
            return 1
        return 0

    def _mutant_rec(self, cat: str, lv: str, shape: str, mut, rtv: bool,
                    c: B.Classification) -> dict:
        rec = self._base(cat)
        rec.update({
            "category": cat,
            "kernel": shape,
            "shape": shape,
            "class": mut.klass,
            "paper_category": mut.paper_category,
            # Record identity, made globally unique: the S12 sanitizer records
            # join on this exact string, so it must name the RTV mode too (S12
            # measures the bare pipeline, i.e. rtv=off).
            "mutant_id": f"{self.toolchain}-{cat}-{shape}-{mut.mutant_id}"
                         f"-{'on' if rtv else 'off'}",
            "spec_id": mut.spec_id,
            "level": lv,
            "rtv": "on" if rtv else "off",
            "outcome": c.outcome,
            # homework-check C1: `stage` enum is ["compile","runtime","none"],
            # where "none" is the not-detected value. A `never` outcome means the
            # kernel ran and silently produced a wrong result — nothing fired, so
            # there is no stage to name.
            "stage": c.stage if c.outcome != "never" else "none",
            "manifest": c.manifest or "noop",
            "n_asserts": c.n_asserts,
        })
        d = _detail(c)
        if d:
            rec["detail"] = d
        return rec

    # -- S12: AddressSanitizer --------------------------------------------

    def cmd_s12(self, size: str = "small") -> int:
        """External-checker supplement: what ASan catches on the BARE pipeline.

        `rtv=False` is deliberate. RTV's own `cf.assert` bounds checks would
        abort *before* the faulty access, so ASan would see a clean program and
        report nothing. S12 asks a different question from S1: not "did the
        toolchain's generated check fire" but "does an external memory checker
        catch what the bare pipeline lets through". That is the
        `flagged ∧ exercised` delta the manifest's S12 row wants.

        EVERY mutant that produced a kernel gets a record, including the ones
        the verifier rejected at lowering. Those are honest
        `flagged=false, exercised=false` rows with a `detail` naming the
        compile-time rejection — the toolchain caught the defect before any
        memory access could occur, which is a legitimate outcome and not a
        harness failure. Skipping them would make S12's `total` disagree with
        the mutant census and break completeness checking.

        Mutants that are `n/a` (not expressible on this surface) get NO
        sanitizer record: there is no kernel to sanitize. `n_not_expressible` is
        reported separately so the arithmetic still reconciles.
        """
        w = self._writer("sanitizer", "sanitizer")
        klass = self.cfg["klass"]
        tally: Counter = Counter()
        unflagged: Counter = Counter()
        uinstr: dict[tuple[str, str], list[int]] = defaultdict(list)
        n_na = 0

        # Same per-family selection the E1 battery used, or S12's `total` cannot
        # reconcile with the mutant census.
        sel = (select_m2(list(self.cfg["battery_cats"]), self.cfg["specs"],
                         size=size)
               if klass == "M2" else None)
        variant = _one_variant_per_injection(self.cfg["specs"], sel)

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # battery_cats, NOT categories: S12 must cover exactly the mutants
            # the E1 battery produced, or its `total` cannot reconcile with the
            # mutant census. On linalg those differ (5 M2 targets vs 7 composed).
            for cat in self.cfg["battery_cats"]:
                for dynamic in (False, True):
                    shape = "dynamic" if dynamic else "static"
                    clean = C.make_case(cat, size=size, dynamic=dynamic)
                    clean_src = self._emit_mutant(clean, clean, None)
                    for mut in self.cfg["specs"]:
                        if klass == "M1" and not m1_cell(cat, mut.spec_id):
                            continue
                        if sel is not None and (cat, shape, mut.spec_id) not in sel:
                            continue
                        if mut is not variant.get((cat, shape, mut.spec_id), mut):
                            continue
                        try:
                            mcase, structural = M.apply(clean, mut)
                            src = self._emit_mutant(mcase, clean, structural)
                            if src == clean_src:
                                raise AssertionError(
                                    "emitter produced a module byte-identical "
                                    "to the clean kernel -- the perturbation "
                                    "is invisible to this surface")
                        except (M.NotApplicable, L.NotExpressible) as exc:
                            n_na += 1
                            log(f"  {cat:22s} {shape:8s} {mut.mutant_id:16s} "
                                f"n/a — not sanitized ({exc})")
                            continue
                        except AssertionError as exc:
                            n_na += 1
                            log(f"  {cat:22s} {shape:8s} {mut.mutant_id:16s} "
                                f"NOOP mutation — not sanitized ({exc})")
                            continue

                        work = tmp / f"{cat}_{shape}_{mut.mutant_id}"
                        work.mkdir(parents=True, exist_ok=True)
                        f = work / "m.mlir"
                        f.write_text(src)
                        r = (B.run_sanitizer_gpu(f, work, self.surface, rtv=False)
                             if self.cfg.get("sanitizer") == "compute-sanitizer"
                             else B.run_asan(f, work, self.surface, rtv=False))
                        rec = r.as_record(self.toolchain, cat, klass,
                                          f"{self.toolchain}-{cat}-{shape}-"
                                          f"{mut.mutant_id}-off")
                        rec["settings_hash"] = self.settings.get(cat, "MISSING")
                        rec["toolchain_version"] = self.version
                        rec["shape"] = shape
                        rec["rtv"] = "off"
                        rec["instrumented"] = str(r.instrumented)
                        # Which link of lower->translate->opt->clang->run this
                        # record stopped at. Needed to tell the two kinds of
                        # zero-coverage record apart in `stats`: stage=instrument
                        # means ASan inserted no checks (no measurement obtained,
                        # a harness defect), stage=lower means the verifier
                        # rejected the mutant so there was never a binary.
                        rec["asan_stage"] = r.stage
                        w.write(rec)
                        tally[(rec["flagged"], rec["fault"], rec["exercised"])] += 1
                        # Record WHY each miss happened, keyed by spec — but ONLY
                        # for mutants that actually ran with coverage. A
                        # verifier-rejected mutant (instrumented=0, exercised=
                        # false) was never a candidate for a sanitizer report, so
                        # counting it as a "miss" would misrepresent it: the
                        # toolchain caught it before any memory access. The
                        # misses that matter are the ones that ran clean under
                        # real ASan coverage — the defects a memory checker
                        # structurally cannot see.
                        if rec["flagged"] != "true" and r.instrumented > 0:
                            unflagged[(mut.spec_id, shape)] += 1
                            uinstr[(mut.spec_id, shape)].append(r.instrumented)
                        log(f"  {cat:22s} {shape:8s} {mut.mutant_id:16s} "
                            f"instr={r.instrumented:3d} stage={r.stage:11s} "
                            f"flagged={rec['flagged']:5s} fault={rec['fault']:11s} "
                            f"exercised={rec['exercised']:5s} | {r.detail[:48]}")

        # Reconcile against the committed census rather than trusting the loop
        # above. Every non-n/a rtv=off mutant record must have exactly one
        # sanitizer record; a silent disagreement here is the completeness
        # failure this command exists to prevent.
        mutants = self._load("mutants.jsonl")
        off = [m for m in mutants if m.get("rtv") == "off"
               and m.get("outcome") != "n/a"]
        san_records = self._load("sanitizer.jsonl")
        missing = sorted({m["mutant_id"] for m in off}
                         - {r["mutant_id"] for r in san_records})
        problems = []
        if missing:
            problems.append(
                f"{len(missing)} mutant(s) have no sanitizer record: "
                + ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else ""))
        if not mutants:
            problems.append("mutants.jsonl is empty — run `minimal` before `s12`")

        # Reconcile the miss breakdown against the tally. An earlier revision
        # keyed `unflagged` on (spec, shape, instrumented) but serialized on
        # f"{spec}/{shape}", so the five distinct coverage counts per cell
        # collided and four of every five misses were silently dropped — stats
        # then reported n=1 where the true count was n=5. The sum must equal the
        # number of records that ran clean under real coverage.
        n_ran_clean = sum(1 for r in san_records
                          if r.get("exercised") == "true"
                          and r.get("flagged") != "true"
                          and int(r.get("instrumented", "0") or 0) > 0)
        if sum(unflagged.values()) != n_ran_clean:
            problems.append(
                f"miss breakdown sums to {sum(unflagged.values())} but "
                f"{n_ran_clean} mutants ran clean under real sanitizer coverage")

        (self.raw / "s12-census.json").write_text(json.dumps({
            "class": klass,
            "size": size,
            "n_sanitized": sum(tally.values()),
            "n_not_expressible": n_na,
            "n_mutants_rtv_off": len(off),
            "reconciled": not missing,
            "tally": {f"flagged={f}/fault={fa}/exercised={e}": n
                      for (f, fa, e), n in sorted(tally.items())},
            # Which specs the external checker MISSED, and with what coverage.
            # Every entry here ran with instrumented > 0 (a zero would be a false
            # negative and the stats gate rejects the run), so each miss is a
            # genuine "the faulting access never left the allocation" case, not
            # an instrumentation failure. This is the S12 residue the paper
            # reports: what a memory checker structurally cannot see. The
            # instrumented range is carried so a reader can confirm real coverage
            # backed every miss.
            "unflagged_by_spec": {
                f"{sid}/{shape}": {
                    "n": n,
                    "instrumented_min": min(uinstr[(sid, shape)]),
                    "instrumented_max": max(uinstr[(sid, shape)]),
                }
                for (sid, shape), n in sorted(unflagged.items())},
            "problems": problems,
        }, indent=2) + "\n")

        if w.errors:
            log(f"[{self.toolchain}] s12: {len(w.errors)} SCHEMA ERROR(S)")
            for e in w.errors[:10]:
                log(f"  !! {e}")
            return 1
        if problems:
            log(f"[{self.toolchain}] s12: {len(problems)} PROBLEM(S)")
            for p in problems:
                log(f"  !! {p}")
            return 1
        log(f"[{self.toolchain}] s12: {sum(tally.values())} sanitized, "
            f"{n_na} not expressible, reconciled with {len(off)} rtv=off mutants")
        return 0

    # -- E3: S8 expressibility + S9 remainder ------------------------------

    def cmd_e3(self, size: str = "small") -> int:
        """S8 (expressibility) and S9 (remainder), both DERIVED FROM MEASUREMENT.

        S8 is not asserted from prose. Two of the four obligation classes are
        measured here; the other two are structural facts about the surface:

        * `elem`  — MEASURED. Does RTV generate a bounds guard on the clean
                    kernel's own accesses (`loc("kernel")`, oracle excluded)?
        * `shape` — MEASURED. Does the verifier reject a shape-contract mutant at
                    lowering, or does RTV emit an extent/subview diagnostic?
        * `loop`  — no. No loop-bound or trip-count diagnostic class appears in
                    any RTV output on either surface (measured across all
                    composed categories and both shape modes).
        * `hw`    — no. No hardware-constraint construct appears in any RTV
                    output on either surface, so there is no hardware contract
                    for a guard to discharge; M3 is n/a on both lanes.

        S9 counts KERNEL guards only — see rule 2 in the module docstring.
        """
        # ---- measure elem + shape per category ---------------------------
        elem_yes: dict[str, bool] = {}
        shape_diag: dict[str, bool] = {}
        kernel_guards: dict[str, int] = {}
        shape_classes: Counter = Counter()

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for cat in self.cfg["categories"]:
                tot = 0
                for dynamic in (False, True):
                    shape = "dynamic" if dynamic else "static"
                    src = self._emit_clean(cat, dynamic, size)
                    work = tmp / f"{cat}_{shape}"
                    work.mkdir(parents=True, exist_ok=True)
                    f = work / "k.mlir"
                    f.write_text(src)

                    # RTV-only IR: cf.assert counts attributed by location. This
                    # is the stage where the op still exists, so it is where the
                    # guard count is meaningful.
                    rtv_ir = work / "rtv.mlir"
                    r = B.run_mlir_opt(B.RTV_ONLY[self.surface], f, rtv_ir)
                    if not r.ok:
                        log(f"  {cat:22s} {shape:8s} RTV-PIPELINE-FAIL")
                        continue
                    n_kernel, _n_total = B.count_kernel_asserts(rtv_ir)
                    tot += n_kernel
                    elem_yes[cat] = elem_yes.get(cat, False) or n_kernel > 0

                    # Fully lowered RTV-on IR: decode the assert payloads to see
                    # WHICH KIND of obligation each guard discharges.
                    ll = work / "k.ll.mlir"
                    if B.run_mlir_opt(B.pipeline(self.surface, True), f, ll).ok:
                        for m in B.assert_messages(ll):
                            lm = B._LOC_TAG.search(m)
                            if lm and lm.group(1) == "oracle":
                                continue          # harness artifact, not kernel
                            diag = next((ln.strip() for ln in m.splitlines()
                                         if ln.strip().startswith("^")), "")
                            # Normalize the SSA names and dimension numbers so
                            # identical diagnostics collapse into one class:
                            # that is what distinguishes a *bounds* guard from an
                            # extent/subview one. (Named `norm` to avoid
                            # shadowing the mutation-class `klass`.)
                            norm = re.sub(r"\d+", "N", diag)
                            shape_classes[norm] += 1
                            # An extent/subview/offset diagnostic is a SHAPE
                            # obligation, not a plain element-bounds one.
                            if re.search(r"subview|offset|extent|dimension",
                                         norm):
                                shape_diag[cat] = True
                kernel_guards[cat] = tot

        # A surface expresses the shape obligation if its verifier rejects shape
        # mutants at compile time. Read that off the committed mutant records
        # rather than re-running the battery.
        shape_rejected = self._shape_rejections()

        ew = self._writer("expressibility", "expressibility")
        shape_evidence: dict[str, str] = {}
        for cat in sorted(self.cfg["categories"]):
            elem = "yes" if elem_yes.get(cat) else "no"
            diag = bool(shape_diag.get(cat))
            rejected = shape_rejected.get(cat, 0) > 0
            shape = "yes" if (diag or rejected) else "no"
            # The two shape signals have different coverage, and flattening them
            # would overstate the evidence. A composed category that is NOT a
            # mutation-battery target (linalg relu/transpose: composed and gated
            # by E2, but not one of §5's five M2 categories) never had a shape
            # mutant injected, so its verdict rests on the RTV diagnostic alone.
            # That is still a measurement — the clean kernel's guards emit only
            # `^ out-of-bounds access`, no extent/subview class — but it is a
            # weaker one, and it is recorded as such rather than silently
            # counted equal to a category with both signals.
            shape_evidence[cat] = ("mutation+diagnostic" if (diag and rejected)
                                   else "mutation-only" if rejected
                                   else "diagnostic-only" if diag
                                   else "diagnostic-only (no shape diagnostic)")
            for cls, expr in (("elem", elem), ("shape", shape),
                              ("loop", "no"), ("hw", "no")):
                rec = {"toolchain": self.toolchain,
                       "toolchain_version": self.version,
                       "settings_hash": self.settings.get(cat, "MISSING"),
                       "category": cat, "class": cls, "expressible": expr}
                ew.write(rec)

        rw = self._writer("remainder", "remainder")
        for cat in sorted(self.cfg["categories"]):
            rw.write({
                "toolchain": self.toolchain,
                "toolchain_version": self.version,
                "settings_hash": self.settings.get(cat, "MISSING"),
                "category": cat,
                "unconditional_guards": kernel_guards.get(cat, 0),
                "criterion_ref": ("RTV-generated cf.assert at loc(kernel), summed "
                                  "over the category's static+dynamic clean "
                                  "kernels; loc(oracle) guards excluded"),
            })

        (self.raw / "e3-detail.json").write_text(json.dumps({
            "size": size,
            # S9 is size-invariant on `linalg` but NOT on `low`: the low kernels
            # are hand-tiled, so RTV's guard count tracks loop-nest depth, and
            # FULL_DIMS raises relu/softmax from rank 2 to rank 4. `low` measures
            # 108 at small size and 132 at full. `remainder.jsonl` carries no size
            # field, so this is the only place a reader can tell which size the
            # committed S9 came from.
            "size_caveat": ("S9 depends on the shape table on the `low` surface "
                            "(hand-tiled loop nests); the committed figure is the "
                            "one measured at the size recorded above"
                            if self.surface == "low" else
                            "S9 is size-invariant on this surface: a "
                            "linalg.generic's instrumented access count is fixed "
                            "by its indexing maps, not by tensor rank"),
            "kernel_guards_per_category": kernel_guards,
            "elem_measured": elem_yes,
            "shape_diagnostics": shape_diag,
            "shape_compile_rejections": shape_rejected,
            "shape_evidence_basis": shape_evidence,
            "assert_diagnostic_classes": dict(shape_classes.most_common()),
        }, indent=2) + "\n")

        errs = ew.errors + rw.errors
        if errs:
            log(f"[{self.toolchain}] e3: {len(errs)} SCHEMA ERROR(S)")
            for e in errs[:10]:
                log(f"  !! {e}")
            return 1
        log(f"[{self.toolchain}] e3: {len(self.cfg['categories'])} categories x "
            f"4 obligation classes = {len(self.cfg['categories']) * 4} "
            f"expressibility rows; Σ kernel guards = {sum(kernel_guards.values())}")
        log(f"    assert diagnostic classes: {dict(shape_classes.most_common())}")
        return 0

    def _shape_rejections(self) -> dict[str, int]:
        """Per category, how many mutants the verifier rejected at lowering."""
        p = self.raw / "mutants.jsonl"
        out: dict[str, int] = defaultdict(int)
        if not p.exists():
            return {}
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("outcome") == "compile":
                out[r["category"]] += 1
        return dict(out)

    # -- collect / stats ---------------------------------------------------

    def _load(self, name: str) -> list[dict]:
        p = self.raw / name
        if not p.exists():
            return []
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    def cmd_collect(self) -> int:
        """raw/*.jsonl -> results/<lane>/*.jsonl, re-validating every record.

        Idempotent: rewrites each stream from raw rather than appending, so a
        re-run cannot duplicate records (homework-check T3 caught exactly that in
        the triton lane's expressibility stream).
        """
        self.results.mkdir(parents=True, exist_ok=True)
        n = 0
        errs: list[str] = []
        kinds = {"kernels.jsonl": "kernel", "mutants.jsonl": "mutant",
                 "sanitizer.jsonl": "sanitizer",
                 "expressibility.jsonl": "expressibility",
                 "remainder.jsonl": "remainder"}
        for name, kind in kinds.items():
            rows = self._load(name)
            if not rows:
                continue
            for r in rows:
                errs.extend(B.validate(r, kind, self.schema))
            (self.results / name).write_text(
                "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
            n += len(rows)
        # Carry the census files across too: they are the audit trail for the
        # arithmetic asserted in stats.json.
        for name in ("setup.json", "minimal-census.json", "s12-census.json",
                     "e3-detail.json"):
            p = self.raw / name
            if p.exists():
                (self.results / name).write_text(p.read_text())
        if errs:
            log(f"[{self.toolchain}] collect: {len(errs)} SCHEMA ERROR(S)")
            for e in errs[:20]:
                log(f"  !! {e}")
            return 1
        log(f"[{self.toolchain}] collect: {n} records -> {self.results} "
            f"(all schema-valid)")
        return 0

    def cmd_stats(self) -> int:
        kernels = self._load("kernels.jsonl")
        mutants = self._load("mutants.jsonl")
        san = self._load("sanitizer.jsonl")
        expr = self._load("expressibility.jsonl")
        rem = self._load("remainder.jsonl")
        klass = self.cfg["klass"]
        lane = self.cfg["lane"]

        # ---- S1: detection matrix, per class, RTV modes broken out --------
        # Aggregate at INJECTION level (spec_id), not record level: any variants
        # a spec declares are collapsed onto their shared injection identity.
        #
        # The class a record is filed under is resolved from its spec_id through
        # the taxonomy (`_attr_class`), NOT read from the record's stamped
        # `class`. A re-homed spec carries its new family even where the writer
        # still stamps the old class (M1.6 -> M4-d), so reading the stamp would
        # file the 8 M4-d injections under M1 and leave this lane's M4 cell
        # empty when its instances exist.
        inj: dict[tuple[str, str, str, str, str], list[str]] = defaultdict(list)
        for r in mutants:
            inj[(_attr_class(r), r["rtv"], r["category"], r["shape"],
                 r.get("spec_id", r["mutant_id"]))].append(r["outcome"])

        per_cls: dict[str, dict[str, Counter]] = {
            rtv: defaultdict(Counter) for rtv in ("off", "on")}
        for (cls, rtv, _cat, _shape, _spec), outs in inj.items():
            per_cls[rtv][cls][_reduce_injection(outs)] += 1

        def row(tally: Counter) -> dict:
            return {"n_injected": sum(tally.values()),
                    "n_compile": tally.get("compile", 0),
                    "n_runtime": tally.get("runtime", 0),
                    "n_never": tally.get("never", 0),
                    "n_na": tally.get("n/a", 0)}

        # Every class the axis marks `measured` for this lane gets a real cell,
        # built from that class's own injections. The lane's own `klass` is no
        # longer the only measured class (M4 is measured on both surfaces too),
        # so the matrix is driven by the axis, not by a single literal.
        status = AX.lane_status(lane)
        s1: dict[str, dict] = {}
        uncompared: list[str] = []
        for cls in MUTATION_CLASSES:
            if status.get(cls) == "measured":
                on = per_cls["on"].get(cls, Counter())
                off = per_cls["off"].get(cls, Counter())
                r = row(on)
                r["by_rtv"] = {"off": row(off), "on": row(on)}
                r["note"] = (
                    f"measured on the {self.surface} surface. Top-level counts "
                    f"are RTV-on (generate-runtime-verification enabled), "
                    f"because a detection matrix asks what the toolchain can "
                    f"catch and RTV is part of MLIR; `by_rtv.off` is the bare "
                    f"pipeline, which is the baseline S12's delta is computed "
                    f"against. The two modes are separate measurements and are "
                    f"never merged (manifest §5.1). n_injected counts "
                    f"INJECTIONS (spec level), not records: the whole lane's "
                    f"{len(mutants)} mutant records reduce to "
                    f"{r['n_injected']} {cls} injections.")
                s1[cls] = r
            elif cls in _na_classes(lane):
                why = AX.na_reason(lane, cls)
                # homework-check I2: a class the surface cannot express must be
                # counted as n/a with n_na = N, never reported as 0 while the
                # note claims n/a.
                s1[cls] = {"n_injected": 0, "n_compile": 0, "n_runtime": 0,
                           "n_never": 0, "n_na": C.N_TARGET_PER_CLASS,
                           "note": f"no expressible {cls} mutant on this "
                                   f"surface; all spec N="
                                   f"{C.N_TARGET_PER_CLASS} mutants => n/a. {why}"}
            else:
                # UNCOMPARED, not n/a. This lane has never been run against the
                # class, so emitting a cell -- even a zero -- would report an
                # unmeasured gap as a measurement, and calling it `n/a` would
                # report it as a measured capability limit. No cell; the class
                # is declared in `S1_declared_uncompared` below so the omission
                # is a stated scope decision rather than a missing iteration.
                uncompared.append(cls)

        # S1's counterpart to `AWP`'s all-n/a row: the classes this lane was
        # never run against, named rather than left out. `n/a` and `uncompared`
        # differ (see AX.status_meaning) and a reader must not have to infer
        # which one a missing key means.
        s1_declared_uncompared = {
            "classes": sorted(uncompared),
            "reason": AX.uncompared_reason(),
            "note": ("No cell is emitted for these classes. A missing cell is "
                     "NOT an n/a cell: `n/a` means this surface was measured and "
                     "cannot express the defect, while a missing cell means the "
                     "lane was never run against the class."),
        } if uncompared else {}

        # ---- S8: 4 obligation classes x {yes, partial, no} ----------------
        s8 = {cls: {"yes": 0, "partial": 0, "no": 0}
              for cls in AX.obligation_classes()}
        for r in expr:
            s8[r["class"]][r["expressible"]] += 1

        # ---- S9: remainder, kernel guards only ----------------------------
        s9_per = {r["category"]: r["unconditional_guards"] for r in rem}

        # ---- S12: sanitizer supplement ------------------------------------
        s12: dict[str, dict] = {}
        stale: list[str] = []
        for r in san:
            c = _attr_class(r)
            d = s12.setdefault(c, {"flagged_and_exercised": 0, "total": 0,
                                   "flagged": 0, "exercised": 0,
                                   "not_instrumented": 0,
                                   "rejected_before_run": 0})
            d["total"] += 1
            if r.get("flagged") == "true":
                d["flagged"] += 1
            if r.get("exercised") == "true":
                d["exercised"] += 1
            if r.get("flagged") == "true" and r.get("exercised") == "true":
                d["flagged_and_exercised"] += 1
            n_instr = int(r.get("instrumented", "0") or 0)
            stage = r.get("asan_stage")
            # TWO different zero-coverage situations, and conflating them would
            # misreport the lane's health. `exercised` CANNOT separate them —
            # run_asan returns early with exercised=False in both — so the stage
            # recorded on the record is what does:
            #
            # * stage="lower" — the verifier rejected the mutant, so there was
            #   never a binary to instrument. Legitimate outcome, counted as
            #   `rejected_before_run`. On linalg M2 this is 34 of 54.
            # * stage="instrument" — the chain reached the IR but `opt
            #   -passes=asan` inserted no `__asan_report_*` call sites, so NO
            #   MEASUREMENT WAS OBTAINED. This is the false negative the gate
            #   exists to catch (see the S12 rationale in this module's
            #   docstring: LLVM's ASan pass only instruments functions carrying
            #   `sanitize_address`, which mlir-translate never emits, so coverage
            #   can silently be zero). Must stay 0.
            #
            # Grepping `__asan_load4` is NOT sufficient — those appear as
            # declarations even when nothing was instrumented. Only the count of
            # `__asan_report_{load,store}` *call sites* distinguishes the two.
            if n_instr == 0:
                if stage is None:
                    # A record written before `asan_stage` existed. Guessing here
                    # would let a genuine instrumentation failure be filed as a
                    # legitimate rejection, so it is reported instead.
                    stale.append(r.get("mutant_id", "?"))
                elif stage == "instrument":
                    d["not_instrumented"] += 1
                else:
                    d["rejected_before_run"] += 1
        for cls in MUTATION_CLASSES:
            # Only classes this lane RAN can contribute a sanitizer row.
            # `uncompared` classes are skipped here for the same reason S1
            # emits no cell for them: "nothing to sanitize" would read as a
            # measured zero. This is the third place the literal 3-class tuple
            # used to be, and the one that mattered most -- an ASan row of
            # (0 flagged, 0 exercised) is indistinguishable from a clean lane.
            if cls in uncompared:
                continue
            s12.setdefault(cls, {"flagged_and_exercised": 0, "total": 0,
                                 "flagged": 0, "exercised": 0,
                                 "not_instrumented": 0,
                                 "rejected_before_run": 0,
                                 "note": "no expressible mutant on this surface "
                                         "=> nothing to sanitize"})

        # ---- kernel gate --------------------------------------------------
        # Keyed by (category, size) because both sizes are committed. Collapsing
        # them into one per-category count would make `composed` read 4 where the
        # suite has 2 shapes, and would hide a full-size-only regression behind a
        # passing small-size row.
        per_cat: dict[str, dict] = {}
        for r in kernels:
            d = per_cat.setdefault(r["category"], {})
            e = d.setdefault(r["size"], {"composed": 0, "compiled": 0,
                                         "ran_ok": 0, "ref_pass": 0,
                                         "rtv_on_ref_pass": 0})
            e["composed"] += 1
            e["compiled"] += r["compile"] == "ok"
            e["ran_ok"] += r.get("run") == "ok"
            e["ref_pass"] += r["ref_check"] == "pass"
            e["rtv_on_ref_pass"] += r.get("rtv_on_ref_check") == "pass"

        # homework-check I3: every non-passing kernel needs a documented,
        # auditable disposition rather than a silent shortfall in `ref_pass`.
        failures = [{"category": r["category"], "kernel": r["kernel"],
                     "size": r["size"], "compile": r["compile"],
                     "run": r.get("run"), "ref_check": r["ref_check"],
                     "detail": r.get("detail", ""),
                     "disposition": ("harness/emitter defect — must be fixed, "
                                     "not accepted" if r["compile"] != "ok"
                                     else "gate failure — see detail")}
                    for r in kernels
                    if r["compile"] != "ok" or r["ref_check"] != "pass"]

        composed = sorted(self.cfg["categories"])
        missing = sorted(set(ALL_CATEGORIES) - set(composed))
        census = (json.loads((self.raw / "minimal-census.json").read_text())
                  if (self.raw / "minimal-census.json").exists() else {})
        s12_census = (json.loads((self.raw / "s12-census.json").read_text())
                      if (self.raw / "s12-census.json").exists() else {})

        # Surface WHICH specs the external checker missed. The misses are the
        # interesting half of S12 — they are the defects a memory checker
        # structurally cannot see, which is the residue the benchmark exists to
        # measure. Reporting only the aggregate `flagged_and_exercised` would
        # hide the distinction between "ASan found nothing because nothing was
        # wrong" and "ASan found nothing because the fault never left the
        # allocation". Every entry carries instrumented > 0, so each is the
        # latter, not an instrumentation failure.
        if s12_census.get("unflagged_by_spec"):
            s12.setdefault(klass, {})["misses"] = s12_census["unflagged_by_spec"]
            s12[klass]["miss_note"] = (
                "each of these ran with real sanitizer coverage (instrumented > 0) "
                "and still produced no report: the injected defect corrupts the "
                "result WITHOUT any access leaving its allocation, so no memory "
                "checker can see it. This residue — not the flagged count — is "
                "what S12 measures.")

        # The S12 false-negative gate is a HARD failure, not a reported number.
        # A zero-coverage record means the chain produced no measurement at all,
        # so the lane's S12 result is untrustworthy and the run must be reported
        # as not-green. This is the gate that catches the four measured ASan
        # defects silently regressing (see the S12 rationale in this module's
        # docstring).
        for cls, d in s12.items():
            if d.get("not_instrumented"):
                failures.append({
                    "category": "*", "kernel": "*",
                    "size": sorted({r["size"] for r in kernels}) or ["unknown"],
                    "compile": "ok", "run": "ok", "ref_check": "pass",
                    "detail": (f"S12 {cls}: {d['not_instrumented']} mutant(s) "
                               f"reached the IR but got 0 ASan check sites — no "
                               f"measurement was obtained"),
                    "disposition": ("ASan instrumentation defect — must be fixed, "
                                    "not accepted. Check that opt -passes=asan ran, "
                                    "that the sanitize_address attribute was "
                                    "attached, and that the target triple/"
                                    "datalayout were injected."),
                })
        if stale:
            failures.append({
                "category": "*", "kernel": "*",
                "size": sorted({r["size"] for r in kernels}) or ["unknown"],
                "compile": "ok", "run": "ok", "ref_check": "pass",
                "detail": (f"S12: {len(stale)} sanitizer record(s) have 0 "
                           f"coverage and no `asan_stage`, so a legitimate "
                           f"compile-time rejection cannot be told from an "
                           f"instrumentation failure: "
                           + ", ".join(stale[:5])
                           + (" ..." if len(stale) > 5 else "")),
                "disposition": ("stale s12 output — re-run `./run.sh s12` with "
                                "the current lane.py, which records asan_stage"),
            })

        stats = {
            "toolchain": self.toolchain,
            "toolchain_version": self.version,
            "surface": self.surface,
            "device": (f"{self.backend} " +
                       (f"({B.cuda_chip()}; kernels JIT-compiled to a cubin and "
                        "run on the device)" if self.gpu else
                        "(JIT on host; no GPU dependency)")),
            "S1_detection": s1,
            "S1_declared_uncompared": s1_declared_uncompared,
            "S1_class_axis": {
                "source": "schema/class-axis.json",
                "axis_version": AX.axis()["axis_version"],
                "mutation_classes": MUTATION_CLASSES,
                "status": AX.lane_status(self.cfg["lane"]),
                "note": ("The per-class status this lane claims. `measured` "
                         "classes have real cells in S1_detection; `n/a` classes "
                         "have cells with n_na = N_TARGET_PER_CLASS; "
                         "`uncompared` classes have NO cell and are listed in "
                         "S1_declared_uncompared. A reader must be able to tell "
                         "an unmeasured gap from a measured capability limit."),
            },
            "S8_expressibility": s8,
            "S9_remainder": {
                "per_category": {c: s9_per[c] for c in sorted(s9_per)},
                "total": sum(s9_per.values()),
                "criterion_ref": ("RTV-generated cf.assert at loc(kernel); "
                                  "loc(oracle) guards excluded (folding them in "
                                  "inflated S9 by ~17% overall, ~67% on relu)"),
                "note": ("never summed with the other MLIR lane's remainder "
                         "(mutation-specs.md §7): the low lane's kernels are "
                         "hand-tiled with an explicit boundary guard, so their "
                         "guard structure is not comparable"),
            },
            "S12_sanitizer_supplement": s12,
            "S12_method": (
                {
                    "tool": "compute-sanitizer --tool memcheck (CUDA)",
                    "chain": ("mlir-opt <bare GPU pipeline: lower-affine, then "
                              "gpu-lower-to-nvvm-pipeline> | compute-sanitizer "
                              "--tool memcheck --error-exitcode 99 mlir-runner"),
                    "rtv": "off (RTV's own cf.assert fires on the device before "
                           "the faulty access and would make memcheck silent)",
                    "false_negative_gate": ("the lowered module's device launch "
                                            "count is recorded; 0 launches => "
                                            "stage=instrument and never a "
                                            "'clean' verdict"),
                    "why_not_asan": ("the kernel runs on the device, so a "
                                     "natively-linked host ASan binary cannot "
                                     "observe the access; memcheck is the "
                                     "checker that shares the execution model"),
                }
                if self.cfg.get("sanitizer") == "compute-sanitizer" else
                {
                    "tool": "LLVM AddressSanitizer (native, host CPU)",
                    "chain": ("mlir-opt <bare pipeline> | mlir-translate "
                              "--mlir-to-llvmir | patch sanitize_address + target "
                              "triple/datalayout | opt -passes=asan | clang "
                              "-fsanitize=address"),
                    "rtv": "off (RTV's own cf.assert would abort before the faulty "
                           "access and make ASan silent)",
                    "asan_options": B.ASAN_ENV.get("ASAN_OPTIONS", ""),
                    "false_negative_gate": ("every binary's __asan_report_* call "
                                            "sites are counted; 0 sites => "
                                            "stage=instrument and never a 'clean' "
                                            "verdict"),
                }
            ),
            "kernel_gate": {c: per_cat[c] for c in sorted(per_cat)},
            "gate_failures": failures,
            "totals": {
                "kernels": len(kernels),
                "compiled": sum(1 for r in kernels if r["compile"] == "ok"),
                "ran_ok": sum(1 for r in kernels if r.get("run") == "ok"),
                "ref_pass": sum(1 for r in kernels if r["ref_check"] == "pass"),
                "mutant_records": len(mutants),
                "sanitizer_records": len(san),
                "expressibility_records": len(expr),
                "gate_failures": len(failures),
            },
            "census": {
                "class": klass,
                "n_injected": census.get("n_injected"),
                "expected_injected": census.get("expected_injected"),
                "n_records": census.get("n_records"),
                "n_target_per_class": C.N_TARGET_PER_CLASS,
                "delta_vs_target": census.get("delta_vs_target"),
                "by_rtv": census.get("by_rtv"),
                # The mutation battery and the sanitizer supplement are measured
                # at ONE size (recorded here), while the kernel gate spans both
                # small and full. A reader comparing `totals.kernels` (both
                # sizes) against `census.n_records` (one size) needs this to
                # avoid reading the difference as a shortfall.
                "battery_size": census.get("size"),
                "s12_size": s12_census.get("size"),
                "note": ("§5.1's enumeration arithmetic is owner-approved and "
                         "must NOT be trimmed to hit N=40 exactly; the overshoot "
                         "is recorded here rather than silently absorbed"),
            },
            # The user's scope ruling: ship green at the composed subset and
            # document the breadth gap. This is the I3 "documented disposition"
            # pattern applied to scope rather than to a failing kernel.
            "scope": {
                "categories_composed": composed,
                "n_composed": len(composed),
                "categories_missing": missing,
                "n_missing": len(missing),
                "manifest_total": len(ALL_CATEGORIES),
                "note": (f"manifest.md §1 counts {len(ALL_CATEGORIES)} "
                         f"categories; this lane composes {len(composed)}. S8 and "
                         f"the kernel gate are emitted over the composed subset "
                         f"ONLY — the {len(missing)} missing categories "
                         f"({', '.join(missing)}) have no emitter on this "
                         f"surface yet, so their expressibility is UNMEASURED, "
                         f"not 'no'. Fabricating a 'no' row for them would "
                         f"understate the SOTA baseline. FLAGGED FOLLOW-UP for "
                         f"the coordinator: S8's denominator here is "
                         f"{len(composed)}, not {len(ALL_CATEGORIES)}, so it is "
                         f"not directly comparable to the triton/iree rows whose "
                         f"denominator is 15."),
            },
        }
        (self.results / "stats.json").write_text(
            json.dumps(stats, indent=2) + "\n")

        log(f"[{self.toolchain}] stats -> {self.results / 'stats.json'}")
        log(f"    totals: {json.dumps(stats['totals'])}")
        log(f"    S1 {klass}: {json.dumps({k: v for k, v in s1[klass].items() if k != 'note' and k != 'by_rtv'})}")
        log(f"    S12: {json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'note'} for k, v in s12.items()})}")
        log(f"    S8: {json.dumps(s8)}")
        log(f"    S9 total: {stats['S9_remainder']['total']}")
        if failures:
            log(f"[{self.toolchain}] stats: {len(failures)} GATE FAILURE(S) — "
                f"lane is NOT green")
            for f in failures[:10]:
                log(f"  !! {f['category']}/{f['kernel']}: {f['detail'][:100]}")
            return 1
        return 0


def _reduce_injection(outcomes: list[str]) -> str:
    """Collapse one injection's per-variant outcomes into a single S1 outcome.

    An injection is `n/a` only when EVERY variant is inexpressible (§6). Otherwise
    it counts at its MOST-detected variant. An ordinary injection carries a single
    outcome (variants are dealt across hosts), but the reduction still applies if
    a set of variants ever shares one injection. Detection strength runs compile
    > runtime > never, so a spec the verifier rejects in one variant is not washed
    out by a silently-corrupting sibling.
    """
    if not outcomes:
        return "n/a"
    if all(o == "n/a" for o in outcomes):
        return "n/a"
    return max((o for o in outcomes if o != "n/a"),
               key=lambda o: _STRENGTH.get(o, 0))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--surface", required=True, choices=sorted(SURFACES),
                    help="linalg = tensor/linalg entry surface (M2); "
                         "low = memref/affine surface (M1)")
    ap.add_argument("cmd", choices=["setup", "minimal", "e2", "e3", "s12",
                                    "collect", "stats", "all"])
    ap.add_argument("--small", dest="size", action="store_const", const="small",
                    default="small")
    ap.add_argument("--full", dest="size", action="store_const", const="full")
    ap.add_argument("--level2", action="store_true",
                    help="add mutation-specs.md §5's level-2 categories")
    ap.add_argument("--device", default="cpu",
                    help="accepted for run.sh.template compatibility; the lane's "
                         "backend is chosen per surface (mlirbench.backend_for: "
                         "cuda for linalg/low)")
    a = ap.parse_args(argv)

    lane = Lane(a.surface)
    lane.raw.mkdir(parents=True, exist_ok=True)
    lane.results.mkdir(parents=True, exist_ok=True)

    if a.cmd == "all":
        steps = [("setup", lambda: lane.cmd_setup()),
                 ("e2", lambda: lane.cmd_e2(a.size)),
                 ("minimal", lambda: lane.cmd_minimal(a.level2, a.size)),
                 ("s12", lambda: lane.cmd_s12(a.size)),
                 ("e3", lambda: lane.cmd_e3(a.size)),
                 ("collect", lambda: lane.cmd_collect()),
                 ("stats", lambda: lane.cmd_stats())]
        failed = []
        for name, fn in steps:
            log(f"\n===== {lane.toolchain} :: {name} =====")
            rc = fn()
            if rc:
                failed.append(name)
        if failed:
            log(f"[{lane.toolchain}] all: FAILED step(s): {', '.join(failed)}")
            return 1
        log(f"[{lane.toolchain}] all: green")
        return 0

    return {"setup": lane.cmd_setup,
            "e2": lambda: lane.cmd_e2(a.size),
            "minimal": lambda: lane.cmd_minimal(a.level2, a.size),
            "s12": lambda: lane.cmd_s12(a.size),
            "e3": lambda: lane.cmd_e3(a.size),
            "collect": lane.cmd_collect,
            "stats": lane.cmd_stats}[a.cmd]()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
