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

### W2 — Replace the lane-wide cap with per-kernel caps

- Drop `--max-local-mem-capacity=2000000` from `run.sh` `CAP`,
  `run_e1.py` `COMPILE_FLAGS`/`ORACLE_FLAGS`, and any E4/E5 flag sets.
- Lane default returns to the compiler default (340 B/thread on sm_120).
- Kernels that still legitimately exceed it (expected: the softmax
  per-thread-row family, possibly a few conv2d accumulators) get a
  **per-kernel** override recorded in a new `choreo/raw/local_cap.json`:
  `{kernel_id: cap_bytes, reason}`; the harness applies the override only
  for that kernel and stamps it into the record's provenance (settings_hash
  input). `raw/toolchain.json` `pinned_flags` must reflect the actual
  invocations (byte-identical discipline, manifest §5).

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
