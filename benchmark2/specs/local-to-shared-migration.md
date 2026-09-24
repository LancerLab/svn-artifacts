# Spec: retire the lane-wide 2 MB local-memory cap (local→shared migration)

> Commissioned 2026-09-24 (owner decision), after a coordinator-side
> investigation of why the choreo lane pins `--max-local-mem-capacity=2000000`
> on every invocation. Owner's framing: in CUDA, cross-thread-reused tile
> data belongs in shared memory, and higher tiling factors are what shared
> memory is for — a per-thread local cap should not be what makes the corpus
> compile.

## Evidence (coordinator probe, 2026-09-24, binary from 8ae43eab)

- The flag raises the **per-thread `local`** hard cap (CUDA local = private,
  global-backed). Default on sm_120/sm_86 = **340 B/thread**
  (`gpu_target.hpp:56-77`: 2× the 170 B best-practice budget). The lane pins
  2,000,000 in `run.sh:48` and in BOTH E1 arms (`run_e1.py:147-155`,
  detector `-rtc=all` and oracle `-rtc=none`). The `run.sh` rationale comment
  is stale: it says "2048-byte default" and cites matmul/1_bert, which was
  already repaired and no longer fails.
- Compile probe of all 317 suite bases **without** the cap: **36 fail** —
  conv2d 16/21, softmax 9/20, matmul 8/20, max_pool2d 3/20. Two patterns:
  1. **Intentional per-thread staging** (softmax "softmax15 pattern"):
     `l1_input = dma.copy input.chunkat(i, p#q, _) => local` — each thread
     stages its own 768-float row (3072 B, 18× budget). Semantically private
     data; `local` is defensible but perf-toxic (guaranteed spill).
  2. **Cross-thread tile staging in local** (conv2d, residual matmul,
     max_pool2d): `l1_A = dma.copy ... => local` inside nested parallel
   scopes — the same pattern the 2026-09-23 repair already moved to
   `=> shared` in 12 matmul kernels ("Group tile buffers live in shared
   memory (one on-chip copy per group)", matmul/1_bert).
- **Shared-capacity failures (found 2026-09-24, full-suite sweep)**: 6 of
  the 36 fail even WITH the pinned 2 MB cap — the failure is the SHARED
  static check, not the local cap ("shared memory OUT OF BOUND ... exceeds
  the limit of 102400 bytes"): conv2d/19_vit (636 KB/block),
  matmul/17_general (288 KB), matmul/19_transformer (520 KB),
  softmax/19_transformer (1 MB), softmax/20_vit (192 KB),
  softmax/2_cnn (196 KB). All six declare per-block shared allocations above
  the 100 KB/SM hardware of BOTH the sm_86 dev host and the sm_120 target —
  physically non-launchable on this lane. All six date to the 2026-06-15
  suite import (e069af9) and are silently absent from the ledger scan
  (stats ingests 310 of 317 kernels; the 7th skip is unreconciled — find it
  during W3). Note the interaction with W4: the old 300 KB sm_120 table let
  softmax/2, softmax/20 and matmul/17 pass the compile-time check on the
  measurement host despite being non-launchable there; the corrected 100 KB
  table rejects all six.
- **E1 channel neutered**: M3.28/M3.18's expected detection is the static
  local-mem check (`memcheck.hpp` `CheckCtMemUsage`, error band = the cap).
  At 2 MB no local tile can trip it — the warn band (170 B) still fires but
  warnings are not a detection outcome. The mutation corpus's LOCAL-band
  specs are structurally compile-undetectable under the pinned flags.
- **Suspect capacity table**: `gpu_target.hpp:69` lists sm_120 SHARED as
  300 KB; CC 12.0 hardware is 228 KB/SM (227 KB/block opt-in). If wrong, the
  SHARED static check (M3.17-class) is over-permissive on the target arch.

## Work items

### W1 — Migrate cross-thread tile staging to shared

For each of the 36 failing bases, move group-reused DMA tile buffers from
`=> local` to `=> shared`, following the matmul/1_bert pattern. Keep
genuinely per-thread data (accumulators like conv2d's
`local f32[M/#q, Cout/#qq] l1_Y`, the softmax per-thread rows) as `local` —
the migration criterion is *cross-thread reuse*, not "no local anywhere".
Record the per-kernel judgment (which buffers moved, which stayed, why) in
the commit message and a short `raw/local_cap.json` (see W2).

Constraint: SHARED is capacity-checked per arch (`gpu_target.hpp:63-79`).
After migration each kernel must compile for `-arch=sm_120` at the default
cap; if a migrated tile set exceeds the SHARED budget, reduce the tile
factor rather than reaching for the local cap.

### W1b — Redesign the six shared-over-capacity bases

**Status: DONE (2026-09-24, owner lane)** — all six bases compile under the
pinned-cap invocation with zero SHARED OUT OF BOUND and pass `basecheck
--check` on BOTH arms (oracle and rtc-all, detector silent).

- matmul/17, matmul/19: k_tile `[32,32,1]/[32,512,1]` → `[32,32,32]`
  (chunked shared tiles à la matmul/1) + `l1_out` local→shared; both compile
  clean at DEFAULT flags (no local-cap entry needed).
- softmax/2, softmax/19, softmax/20: per-thread staging relabelled
  shared→local (semantically private data — the shared label was the
  2026-06-15 authoring bug). Exact LOCAL footprints recorded in
  `raw/local_cap.json` (50176 / 65536 / 24576 B).
- conv2d/19: full rewrite as cooperative streaming GEMM (à la conv2d/20):
  K streamed in 48 chunks of KT=Kw=16 (chunk (c,kh) = flat-k slice
  [c*Kh*Kw+kh*Kw, +16), contiguous in both operands), [14,CT] output tile in
  local (896 B), ~13.6 KB shared/block. Compile-clean at the pinned cap;
  default fails LOCAL only (entry in local_cap.json).

Two latent miscompilations found and worked around in the conv2d/19 rewrite
(both pre-date this spec; the kernel never ran on any host, so they were
invisible): (1) `span_as` merging NON-contiguous dims (e.g.
`span_as(1, K)` over a strided (Cin,Kh,Kw) patch) silently lowers to a
contiguous stride-1 view — wrong data, no diagnostic; (2) same root cause
on the output path (`o.chunkat(...).span_as(CT, 14)` over a stride-196 view
lowered to stride 14). Rule going forward: span_as may only merge
contiguous or extent-1 dims; write output chunks via 4-D transp into the
natural chunkat view (idiom proven in conv2d/14, conv2d/16). Worth a
compiler-side check — candidate ledger obligation, flag to the compiler
owner. Also observed: cooperative tiled_copy pads the copy tile to the
thread grid (e.g. (16,32) over a (7,2) grid lowers as (21,32)), scribbling
past the declared shared buffer into dead space — benign here, worth a
checker look.

Original W1b text:

The six Class-S bases above are NOT relabel jobs: their shared allocations
exceed the hardware, so the fix is a shared-memory tiling redesign —
chunked/block-streamed tiles in the style of the in-flight conv2d/20 edit
(stream `[CT, KT]` tiles global→shared per K-tile instead of staging whole
operands), sized to fit 100 KB/SM with the per-block opt-in ceiling
(99 KB) as the binding constraint. softmax/19's 1 MB per-block buffer
additionally needs the row-chunking treatment, not just tiling. Verify each
redesigned base compiles for sm_120 at the default flags AND still executes
correctly (basecheck) before it re-enters the scan. These six are the
highest-value targets in this spec: they are currently invisible to E2/E3
(absent from the 310-kernel scan), so landing them also closes the
317-vs-310 suite/scan gap the paper must reconcile at the final number sync.

### W2 — Replace the lane-wide cap with per-kernel caps — DONE 2026-09-24

- Dropped `--max-local-mem-capacity=2000000` from every flag set:
  `run.sh` `LEDGER_FLAGS`, `run_e1.py` `COMPILE_FLAGS`/`ORACLE_FLAGS`,
  `run_e2.py` `LEDGER_FLAGS`, `run_e4.py` `FE_FLAGS`/`GEN_FLAGS`,
  `run_e5.py` `BASE_FLAGS`, `analyze_never.py` `LEDGER_FLAGS`/`EXEC_FLAGS`,
  `arch_sweep.py` `LEDGER_FLAGS`, `arch_e5_probe.py` `BASE_FLAGS`.
- Lane default returns to the compiler default.
- **Default-cap probe (2026-09-24, all 317 bases + 170 mutants,
  `/tmp/probe_w2.jsonl`):** exactly 34 bases exceed the default — 10 softmax
  (per-thread row staging), 16 conv2d (per-thread accumulators; the
  dynamic-shape ones keep the former 2000000 lane cap because symbolic local
  usage is not statically bounded), 5 matmul (4 dynamic + 2 small static),
  3 max_pool2d. All 283 other bases pass at the default; the fail set equals
  the exception set, nothing else. Usage measured invariant across the
  ledger/detector/oracle flag arms on the failing set.
- Overrides live in `choreo/raw/local_cap.json` (34 entries, exact measured
  `cap_bytes` + reason) and are applied per-kernel via `local_caps.py`
  (`flags_for(kernel_id)`; mutants inherit their base's entry — the 6 capped
  mutants all resolve to capped bases, measured usage identical).
  `run_e1.py` stamps `local_cap_bytes` into every E1 record's provenance;
  `run.sh`'s `toolchain.json` writer records `local_cap_overrides:
  raw/local_cap.json` with cap-free `pinned_flags` (re-pinned in W3).
  Stale provenance strings in `run_e5.py`/`stats.py` updated.
- Verified: capped kernels compile via the override and fail without it
  (softmax/19 rc=11 at default); uncapped bases compile at the default;
  `basecheck.py` (shared tooling) threads `kernel_id` and passes
  softmax/19 end-to-end (`--check`, both arms earlier).

### W3 — Re-probe and full pipeline

- Probe all 317 bases at the new configuration: the fail set must equal the
  documented per-kernel exception set, nothing else.
- Then regenerate the E1 corpus and rerun the lane (screen → minimal →
  collect → stats). **Sequence after the in-flight post-repair rerun
  settles** (`toolchain_warning` cleared); do not mix pre-/post-cap corpora
  in one manifest.

### W4 — Verify the sm_120 SHARED capacity (croqtile side) — DONE 2026-09-24

Confirmed against the CUDA C Programming Guide "Memory Information per
Compute Capability" (v13.4.2): **CC 12.x = 100 KB per SM, 99 KB per block
opt-in** (consumer Blackwell is in the 8.6/8.9 class, not the 228 KB
datacenter class; 10.0/10.3 = 228, 10.7 = 328). `gpu_target.hpp`'s 300 KB
was wrong 3× over — the SHARED static check would have accepted kernels no
sm_120 block can launch. Fixed on the croqtile `ledger-artifacts` branch
(per the 2026-09-24 host notice: all croqtile changes on `ledger-artifacts`,
never main) as commit `215e17b0` ("choreo: correct the sm_120
shared-memory capacity", sm_120 → 100 KB following the table's per-SM
convention). NOT yet rebuilt into `build-release/choreo` — the rebuild must
be sequenced with the in-flight probe/rerun so a swapped binary cannot mix
toolchains mid-corpus; rebuild + re-pin as part of W3. Note: the table
follows the per-SM convention; the per-block opt-in caps are 1 KB lower
(99/227 KB) — a per-SM check admits that 1 KB of slack, accepted here as
the codebase's existing convention. Side finding left for later: sm_70 is
listed as 48 KB but V100 (CC 7.0) is 96 KB/SM — out of the benchmark's
target set, fix if the arch is ever exercised.

## E1-impact discipline (paper-facing)

Dropping the cap restores the M3.28/M3.18 static error band, so M3 compile
detections may rise and `never` counts fall. Expected and desirable — but
it means the settled numbers change again. When W3 lands:

- `stats.json` `notes[]` must carry a dated line stating the cap change and
  its direction of effect on M3.
- The coordinator will re-sync the paper's E1 cells (another number sweep
  is already owed for the toolchain settle; this folds into it).

## Deliverables

1. Migrated bases + `raw/local_cap.json` (W1/W2).
2. Probe report: default-cap pass/fail per base, before vs after (W3).
3. croqtile PR for the sm_120 capacity if confirmed wrong (W4).
4. Post-migration settled `stats.json` with the dated notes line.

## Coordination

- The tree is dirty with in-flight worker changes; keep this work on top of
  them and do not clobber the rerun.
- croqtile remains PR-only; artifact-repo commits on main are fine.
