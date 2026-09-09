#!/usr/bin/env bash
# benchmark2/choreo/run.sh — choreo lane harness.
#
# Derived from benchmark2/run.sh.template; the subcommand contract is IDENTICAL
# across all workers (manifest §9 / plan §12.1). Runs and collection are
# separate stages:
#
#   run     -> raw artifacts under benchmark2/choreo/raw/
#   collect -> schema-conformant JSON records (schema/record-schema.json)
#   stats   -> benchmark2/results/choreo/stats.json (schema/statistics-manifest.md)
#
# The choreo lane owns S1-S7, S10, S13.
#
# Two conventions this file establishes for the whole benchmark2 tree:
#
#   * GPU exclusivity (manifest §6 / plan §12.6). Correctness lanes (E1/E2/E3)
#     share the GPUs freely and stamp exclusive=false. Timing lanes (E5) must
#     hold benchmark2/.gpu-lock and stamp exclusive=true. `qc` rejects any
#     cost/residue/latency record carrying exclusive=false, so the lock is what
#     makes a timing number admissible.
#
#   * Provenance. Every record carries settings_hash + kernel_hash + the pinned
#     toolchain version (statistics-manifest: non-negotiable cross-cutting
#     fields). `setup` writes raw/toolchain.json once; everything else reads it.
#
set -euo pipefail

TOOLCHAIN="$(basename "$(dirname "$(realpath "$0")")")"   # choreo
HERE="$(cd "$(dirname "$0")" && pwd)"                     # benchmark2/choreo
ROOT="$(cd "$HERE/.." && pwd)"                            # benchmark2/
REPO="$(cd "$ROOT/.." && pwd)"                            # svn-artifacts/
RAW="$HERE/raw"
RESULTS="$ROOT/results/$TOOLCHAIN"
SUITE="$REPO/benchmark/choreo"
DEVICE="${DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"
LOCK="$ROOT/.gpu-lock"
LOCK_STALE_S="${LOCK_STALE_S:-7200}"

# Pinned binary (manifest §3). NOTE: this is relative to the svn-artifacts root,
# i.e. croqtile/build-release/choreo — there is no build-release/ at REPO top.
CHOREO="$REPO/croqtile/build-release/choreo"
CROQTILE="$REPO/croqtile"

# Pinned suite invocation (manifest §3):
#   --stats -es --max-local-mem-capacity=2000000 -t cute
# The cap is required even for UNMUTATED kernels: matmul/1_bert fails to compile
# at the 2048-byte default (exit=4) and compiles cleanly at 2000000.
CAP="--max-local-mem-capacity=2000000"
LEDGER_FLAGS=(--stats -es "$CAP" -t cute)

PY="${PYTHON:-python3}"

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$TOOLCHAIN] $*"; }

need_choreo() {
  [[ -x "$CHOREO" ]] || die "choreo binary not found or not executable:
  $CHOREO
  Build it first, or run: $0 setup"
}

# ---------------------------------------------------------------------------
# GPU lock — the exclusive=true convention
# ---------------------------------------------------------------------------
gpu_lock_acquire() {
  local waited=0
  while ! mkdir "$LOCK" 2>/dev/null; do
    local age=999999
    if [[ -f "$LOCK/info" ]]; then
      local ts; ts="$(sed -n 's/^ts=//p' "$LOCK/info" 2>/dev/null || echo 0)"
      age=$(( $(date +%s) - ts ))
    fi
    if (( age > LOCK_STALE_S )); then
      log "removing stale lock (age ${age}s > ${LOCK_STALE_S}s)"
      rm -rf "$LOCK"; continue
    fi
    if (( waited == 0 )); then
      log "waiting for $LOCK (held by: $(cat "$LOCK/info" 2>/dev/null | tr '\n' ' '))"
    fi
    sleep 10; waited=$(( waited + 10 ))
    (( waited > 21600 )) && die "gave up waiting for GPU lock after ${waited}s"
  done
  {
    echo "ts=$(date +%s)"
    echo "pid=$$"
    echo "device=$DEVICE"
    echo "toolchain=$TOOLCHAIN"
    echo "cmd=$*"
  } > "$LOCK/info"
  log "acquired GPU lock (device $DEVICE, exclusive=true)"
}

gpu_lock_release() {
  if [[ -d "$LOCK" ]] && grep -q "^pid=$$" "$LOCK/info" 2>/dev/null; then
    rm -rf "$LOCK"
    log "released GPU lock"
  fi
}

# Run a timing lane under the lock. Everything after this point stamps
# exclusive=true.
exclusive() {
  gpu_lock_acquire "$@"
  trap gpu_lock_release EXIT INT TERM
  CUDA_VISIBLE_DEVICES="$DEVICE" "$@"
  local rc=$?
  gpu_lock_release
  trap - EXIT INT TERM
  return $rc
}

# ---------------------------------------------------------------------------
# setup — pin the toolchain version (manifest §5)
# ---------------------------------------------------------------------------
cmd_setup() {
  log "setup: pinning toolchain version"
  [[ -d "$CROQTILE" ]] || die "croqtile checkout not found at $CROQTILE"

  if [[ ! -x "$CHOREO" ]]; then
    log "choreo binary absent — attempting release build"
    ( cd "$CROQTILE" && make build-release -j"$(nproc)" ) \
      || die "build failed; build croqtile/build-release/choreo manually"
  fi
  [[ -x "$CHOREO" ]] || die "still no choreo binary at $CHOREO"

  mkdir -p "$RAW"
  local nvcc cuda
  cuda="${CUDA_HOME:-/usr/local/cuda}"
  nvcc="$("$cuda/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | tail -1)"

  # The version is DERIVED FROM THE BINARY, not from `git rev-parse HEAD` of the
  # checkout. manifest §5's wording ("pin `git rev-parse HEAD` at setup") is what
  # allowed the 2026-09-09 drift: the checkout moved to f2f238f while the running
  # binary was still the 680533f build from 4h47m earlier, so records claimed a
  # commit the binary could not have come from. toolchain.py resolves the version
  # from the binary's mtime against the commit history, records checkout_head
  # SEPARATELY, and flags the mismatch. See that module for the full timeline.
  # HERE is passed explicitly: `python3 -` reads the program from stdin, so it has
  # no __file__ and its cwd is wherever run.sh was invoked from. Without this the
  # `import toolchain` below fails depending on the caller's directory.
  $PY - "$RAW/toolchain.json" "$cuda" "$nvcc" "$CAP" "$TOOLCHAIN" "$CHOREO" "$SUITE" "$REPO" "$HERE" <<'PYEOF'
import json, sys, os, datetime
out_path, cuda, nvcc, cap, toolchain, choreo, suite, repo, here = sys.argv[1:10]
sys.path.insert(0, here)
import toolchain as tc

idt = tc.identity()          # memoized; prints the stale-binary warning via describe()
tc.describe(idt)             # surface the warning in the setup log too

def rel(p):
    return os.path.relpath(p, repo)

out = {
  "toolchain": toolchain,
  # version = the commit the BINARY was built from (what the data came from).
  "version": idt["version"],
  "version_basis": idt["version_basis"],
  # checkout_head is recorded SEPARATELY so a disagreement stays visible rather
  # than being silently resolved in favour of the checkout.
  "checkout_head": idt["checkout_head"],
  "checkout_branch": idt["checkout_branch"],
  "checkout_dirty_source_files": idt["checkout_dirty_source_files"],
  "binary_stale_vs_checkout": idt["binary_stale_vs_checkout"],
  "binary": idt["binary"],
  "binary_sha1_12": idt["binary_sha1_12"],
  "binary_mtime": (datetime.datetime.fromtimestamp(idt["binary_mtime"]).isoformat()
                   if idt.get("binary_mtime") else None),
  "cuda_home": cuda,
  "nvcc": nvcc or None,
  # The oracle/checks-off arm uses the owner's pinned spelling. `--disable-runtime-check`
  # is byte-identical to `-rtc=none` (verified on layer_normalization/3_attention:
  # 0 differing normalized lines, 0 runtime_check sites); `-zero-cost` is NOT
  # equivalent (it also drops the CUDA env check). See run_e5.py FLAGS_OFF.
  "pinned_flags": {"ledger":  ["--stats", "-es", cap, "-t", "cute"],
                   "compile": ["-gs", "-t", "cute", "-kt", cap],
                   "oracle":  ["-gs", "-t", "cute", "-kt", cap, "--disable-runtime-check"]},
  "suite": rel(suite),
  "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
}
json.dump(out, open(out_path, "w"), indent=1)
print(json.dumps(out, indent=1))
PYEOF
  log "setup done -> raw/toolchain.json"

  # Sanity: the pinned invocation must work on a known-good kernel.
  local probe="$SUITE/relu/1_bert_32x512x768_32x512x768.co"
  if [[ -f "$probe" ]]; then
    local tmp; tmp="$(mktemp -d)"
    if "$CHOREO" -gs --dump-ledger="$tmp/l.json" "${LEDGER_FLAGS[@]}" "$probe" \
         >"$tmp/c.log" 2>&1; then
      log "probe OK: ledger emitted $(wc -c <"$tmp/l.json") bytes"
    else
      log "probe FAILED — see below"; tail -20 "$tmp/c.log"
    fi
    rm -rf "$tmp"
  fi
}

toolchain_version() {
  if [[ -f "$RAW/toolchain.json" ]]; then
    $PY -c 'import json;print(json.load(open("'"$RAW"'/toolchain.json"))["version"])'
  else
    git -C "$CROQTILE" rev-parse HEAD 2>/dev/null || echo unknown
  fi
}

# ---------------------------------------------------------------------------
# minimal (E1) — mutation-based detection
# ---------------------------------------------------------------------------
cmd_minimal() {
  need_choreo
  log "minimal (E1): mutation-based bug detection, size=$SIZE"
  log "  croqtile @ $(toolchain_version)"

  # 1. Generate the mutant set (40 per class, specs §0/§4).
  local genargs=()
  [[ -n "$LEVEL2" ]] && genargs+=(--level2)
  $PY "$HERE/gen_mutants.py" "${genargs[@]}" || die "mutant generation failed"

  # 2. Calibrate the specs §7 ground-truth oracle on UNMUTATED bases.
  #    Mandatory: some built-in references are themselves wrong (softmax aborts
  #    on correct code under -D__CHECK__), and matmul does not even compile at
  #    the default local-memory cap. Without this step `manifest` is a guess.
  if [[ ! -f "$RAW/oracle_policy.json" || -n "$RECALIBRATE" ]]; then
    $PY "$HERE/calibrate_oracle.py" --jobs "$JOBS" \
      || die "oracle calibration failed"
  else
    log "  reusing existing raw/oracle_policy.json (--recalibrate to redo)"
  fi

  # 3. Classify. E1 is a correctness lane: GPUs are shared, exclusive=false.
  local e1args=(--jobs "$JOBS" --device "$DEVICE")
  [[ -n "$ONLY" ]] && e1args+=(--only "$ONLY")
  [[ -n "$LIMIT" ]] && e1args+=(--limit "$LIMIT")
  CUDA_VISIBLE_DEVICES="$DEVICE" $PY "$HERE/run_e1.py" "${e1args[@]}" \
    || die "E1 classification failed"
}

# ---------------------------------------------------------------------------
# e2 — generation capability: compose + gate all 15 categories
# ---------------------------------------------------------------------------
cmd_e2() {
  need_choreo
  log "e2: obligation ledger over all 15 categories (S3-S6), size=$SIZE"
  CUDA_VISIBLE_DEVICES="$DEVICE" $PY "$HERE/run_e2.py" \
    --jobs "$JOBS" --device "$DEVICE" --size "$SIZE" \
    || die "E2 failed"
}

# ---------------------------------------------------------------------------
# e3 — remainder: what is left after choreo's obligations
# ---------------------------------------------------------------------------
cmd_e3() {
  need_choreo
  log "e3: remainder / unconditional guards (S7), size=$SIZE"
  CUDA_VISIBLE_DEVICES="$DEVICE" $PY "$HERE/run_e3.py" \
    --jobs "$JOBS" --device "$DEVICE" --size "$SIZE" \
    || die "E3 failed"
}

# ---------------------------------------------------------------------------
# e4 — compile cost (choreo only, S10)
# ---------------------------------------------------------------------------
# Exit-code contract shared by run_e4.py and run_e5.py:
#   0  every kernel measured
#   1  lane COMPLETED and wrote a valid artifact, but some kernels were
#      unmeasurable (nvcc/codegen failure). Those are excluded from the median
#      and reported under `failures` — they are NOT counted as zero cost.
#   2  fatal: no choreo binary / precondition not met
#   3  fatal: host contended and --force not given
# rc=1 is a RESULT, not a failure: 2 of 30 kernels do not compile on this build
# (conv2d/10_dynamic and reshape/14_efficientnet — both arch-independent, see
# raw/arch_sweep.json and the E4 log). Treating rc=1 as fatal would abort
# `run.sh all` immediately after a successful E4 and would never reach E5.
timing_lane_rc() {
  local rc=$1 lane=$2
  case "$rc" in
    0) return 0 ;;
    1) log "$lane: completed with unmeasurable kernels (excluded from the median, not counted as zero cost) — artifact is valid"
       return 0 ;;
    *) die "$lane failed (rc=$rc)" ;;
  esac
}

cmd_e4() {
  need_choreo
  log "e4: compile overhead (S10), size=$SIZE — EXCLUSIVE"
  # §12.6 rule 2: qc REJECTS any `cost` record with exclusive=false, and `cost`
  # is E4's record type. So E4 takes the lock even though the prose calls it
  # host-CPU work — an unlocked E4 produces records that cannot survive qc.
  # run_e4.py derives `exclusive` from the lock's actual presence.
  # --jobs is pinned to 1: E4 is a wall-clock lane and run_e4.py measures
  # sequentially regardless, so a higher value would only mislead.
  local rc=0
  exclusive $PY "$HERE/run_e4.py" \
    --jobs 1 --device "$DEVICE" --size "$SIZE" || rc=$?
  timing_lane_rc "$rc" "E4"
}

# ---------------------------------------------------------------------------
# e5 — runtime residue A/B + detection latency (choreo only, S13)
#
# TIMING LANE: must hold the GPU lock and stamp exclusive=true. qc rejects
# cost/residue/latency records with exclusive=false.
# ---------------------------------------------------------------------------
cmd_e5() {
  need_choreo
  log "e5: runtime residue A/B + latency (S13), size=$SIZE — EXCLUSIVE"
  local rc=0
  exclusive $PY "$HERE/run_e5.py" \
    --jobs 1 --device "$DEVICE" --size "$SIZE" || rc=$?
  timing_lane_rc "$rc" "E5"
}

# ---------------------------------------------------------------------------
# e5b — re-measure ONLY the E5b latency half, reusing the banked E5a checkpoint.
# WHY: E5a is ~27 min of GPU-exclusive residue sampling; E5b is cheap. When an
# E5b-only bug is found and fixed (e.g. the 2026-09-09 arm-asymmetry fix that
# made choreo-entry time nvcc-compile+run while the sanitizer timed run-only),
# re-running the whole `e5` would burn the exclusive window for nothing AND
# replace good E5a data with a fresh noisy draw. This takes the lock (E5b is
# still a latency measurement) but passes --skip-e5a so E5a is loaded from
# raw/e5_runtime.e5a.json untouched.
# ---------------------------------------------------------------------------
cmd_e5b() {
  need_choreo
  log "e5b: latency half only (S13), reusing banked E5a — EXCLUSIVE"
  local rc=0
  exclusive $PY "$HERE/run_e5.py" \
    --jobs 1 --device "$DEVICE" --size "$SIZE" --skip-e5a || rc=$?
  timing_lane_rc "$rc" "E5b"
}

# ---------------------------------------------------------------------------
# never — WHY did E1's `never` mutants come back undetected?
#
# Not a lane in the plan's E1-E5 numbering: E1 measures the outcome, this
# explains the ones that came back `never`. It exists as a subcommand because
# an attribution that only I can reproduce is not an attribution.
#
# WITHOUT --execute it is ledger-only (seconds; no nvcc, no GPU) and reports
# C2 as a CANDIDATE. WITH --execute it rebuilds and runs every `never` mutant
# twice at -rtc=all (hoisting on vs off), which is what promotes C2 to a
# confirmed defect — that arm spawns nvcc and runs kernels, so it is a
# correctness lane under §12.6 (discrete outcomes are contention-insensitive)
# but it must NOT be used as a timing measurement.
# ---------------------------------------------------------------------------
cmd_never() {
  need_choreo
  local extra=()
  [[ -n "$EXECUTE" ]] && extra+=(--execute)
  [[ -n "$ONLY" ]] && extra+=(--only "$ONLY")
  # Route the two arms to DIFFERENT files. analyze_never.py defaults --out to
  # raw/never_attribution.json regardless of --execute, so a ledger-only run
  # silently overwrites the executed artifact: the confirmed-hoisting-defect
  # evidence (which cost ~20 s of nvcc per mutant) is replaced by a run whose
  # C2 rows all say "untested". The executed result is the expensive and more
  # specific one, so it keeps the canonical name and ledger-only gets a suffix.
  local out
  if [[ -n "$EXECUTE" ]]; then
    out="$RAW/never_attribution.json"
  else
    out="$RAW/never_attribution_ledgeronly.json"
  fi
  extra+=(--out "$out")
  log "never: attribute E1's \`never\` mutants (execute=${EXECUTE:-no}) -> ${out#$REPO/}"
  CUDA_VISIBLE_DEVICES="$DEVICE" $PY "$HERE/analyze_never.py" "${extra[@]}" \
    || die "never-attribution failed"
}

# ---------------------------------------------------------------------------
# collect / stats
# ---------------------------------------------------------------------------
cmd_collect() {
  log "collect: raw -> schema-conformant records"
  $PY "$HERE/collect.py" --size "$SIZE" || die "collect failed"
}

cmd_stats() {
  log "stats: -> results/$TOOLCHAIN/stats.json (S1-S7, S10, S13)"
  mkdir -p "$RESULTS"
  $PY "$HERE/stats.py" || die "stats failed"
}

cmd_all() {
  cmd_minimal
  cmd_e2
  cmd_e3
  # Ledger-only attribution of E1's `never` mutants. Cheap (no nvcc) and it is
  # the evidence behind whatever the coordinator writes for S2, so it belongs in
  # the pipeline rather than being an ad-hoc step. Pass --execute to `run.sh
  # never` separately for the two-arm probe that confirms the hoisting defect.
  cmd_never
  cmd_e4
  cmd_collect
  cmd_stats
  cmd_e5          # last: it is the only lane that needs the GPU to itself
  cmd_collect
  cmd_stats
}

usage() {
  cat >&2 <<EOF
usage: $0 {setup|minimal|e2|e3|e4|e5|e5b|never|collect|stats|all} [options]

options:
  --small | --full     record size label (default: --small)
  --device <i>         GPU index (default: $DEVICE)
  --level2             minimal: also generate the level-2 widening (specs §5)
  --only <X>           minimal: restrict to M1|M2|M3
                       never:   comma-separated mutant ids
  --limit <N>          minimal: classify only the first N mutants (smoke test)
  --jobs <N>           parallel workers (default: $JOBS)
  --recalibrate        minimal: redo the oracle calibration
  --execute            never: also run the two-arm -rtc=all probe (spawns nvcc)
  -h | --help          this message

lanes:
  setup     pin croqtile HEAD + verify the binary and the pinned flags
  minimal   E1 mutation-based detection (correctness lane, exclusive=false)
  e2        obligation ledger, 15 categories (S3-S6)
  e3        remainder / unconditional guards (S7)
  e4        compile overhead (S10)
  e5        runtime residue A/B + latency (S13) — TIMING, holds .gpu-lock
  e5b       re-measure latency half only, reusing banked E5a — holds .gpu-lock
  never     attribute E1's \`never\` mutants to a cause (not an E-lane; see cmd_never)
  collect   raw -> schema records
  stats     -> results/$TOOLCHAIN/stats.json
EOF
}

# ---------------------------------------------------------------------------
# argument parsing (a while-loop, not `for a in "$@"`: the template's `shift`
# inside a for-loop does not consume the value of --device)
# ---------------------------------------------------------------------------
SIZE="small"
LEVEL2=""
ONLY=""
LIMIT=""
JOBS="${JOBS:-6}"
RECALIBRATE=""
EXECUTE=""

SUB="${1:-}"
[[ -z "$SUB" || "$SUB" == "-h" || "$SUB" == "--help" ]] && { usage; exit 1; }
shift

while (( $# )); do
  case "$1" in
    --small)       SIZE="small";;
    --full)        SIZE="full";;
    --level2)      LEVEL2=1;;
    --recalibrate) RECALIBRATE=1;;
    --execute)     EXECUTE=1;;
    --only)        ONLY="${2:?--only needs a value}"; shift;;
    --only=*)      ONLY="${1#*=}";;
    --limit)       LIMIT="${2:?--limit needs a value}"; shift;;
    --limit=*)     LIMIT="${1#*=}";;
    --jobs)        JOBS="${2:?--jobs needs a value}"; shift;;
    --jobs=*)      JOBS="${1#*=}";;
    --device)      DEVICE="${2:?--device needs a value}"; shift;;
    --device=*)    DEVICE="${1#*=}";;
    -h|--help)     usage; exit 0;;
    *)             echo "ERROR: unknown option: $1" >&2; usage; exit 1;;
  esac
  shift
done

mkdir -p "$RAW" "$RESULTS"
case "$SUB" in
  setup)   cmd_setup;;
  minimal) cmd_minimal;;
  e2)      cmd_e2;;
  e3)      cmd_e3;;
  e4)      cmd_e4;;
  e5)      cmd_e5;;
  e5b)     cmd_e5b;;
  never)   cmd_never;;
  collect) cmd_collect;;
  stats)   cmd_stats;;
  all)     cmd_all;;
  *)       echo "ERROR: unknown subcommand: $SUB" >&2; usage; exit 1;;
esac
