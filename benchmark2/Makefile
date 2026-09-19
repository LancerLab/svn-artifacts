# ============================================================================
# benchmark2 Makefile — EuroSys 2027 artifact orchestration.
#
# Ties the per-lane `run.sh` harnesses together: run each experiment, collect
# per-lane statistics, then merge them into the paper's tables/figures.
#
#   make help           - this help
#   make all            - experiments + stats + render (the full pipeline)
#   make screen         - GATE 2: registry integrity + corpus census (NO run)
#   make experiments    - run every ready lane's `run.sh all`
#   make e1|e2|e3       - run one experiment stage across the ready lanes
#   make e4|e5          - compile-cost / runtime-residue (choreo only, exclusive)
#   make iree|triton|choreo  - run a single lane's full pipeline
#   make <lane>-<stage> - run one stage of one lane (e.g. make triton-stats)
#   make stats          - re-run collect+stats for every ready lane
#   make render         - merge stats.json -> render/ (summary + tables + figures)
#   make tables|figures - only the tables / figures
#   make drift          - re-check the paper's prose against the statistics
#   make guards         - run every schema/check_class_axis.py guard
#   make test-guards    - prove every one of those guards can fail
#   make dashboard      - regenerate DASHBOARD.md (read-only progress view)
#   make worklist       - regenerate the class handoff worklist (read-only)
#   make paper-check    - diff render/ against the paper, copy nothing
#   make paper          - copy the paper's floats into ../eurosys27 (guarded)
#   make clean          - remove render/ output (lane results are NOT touched)
#
# Experiment knobs (any target; passed to the lane harnesses via the env):
#   make choreo-minimal RTC=high N_P1=16 JOBS=8
#   RTC=<entry|low|medium|high>  stamp the -rtc level of the E1 corpus. The
#       curve is a property of the RUN (mutation-specs-v2 §9.6.1), so the level
#       is recorded with the corpus, never inferred from the detection rate.
#       P1 cells are re-run at every level; P3/P4/L take one injection per cell.
#   N_P1=<N>     P1 budget per class      (default 40)
#   N_CELLS=<N>  injections per (spec x category) cell for P3/P4/L (default 1)
#   JOBS=<N>     parallel workers        (default 6)
#   DEVICE=<i>   GPU index               (default $CUDA_VISIBLE_DEVICES or 0)
#   LEVEL2=1     also generate the level-2 category widening
#   ONLY=<x>     restrict the corpus (minimal: M1|M2|M3; never: mutant ids)
#   LIMIT=<N>    classify only the first N mutants (smoke test)
#
# Ready lanes: iree, triton, choreo.
# mlir-linalg and mlir-low also carry committed results/<lane>/stats.json, so
# render.py LOADS them (LANE_STATS_PATHS) and the E1 matrix reproduces the
# paper's 15 rows; they are simply not runnable here (no run.sh harness, no
# extern deps). tilelang has no results/ at all and stays in NOT_READY.
# The four lanes' corpora are frozen and must be JOINED to the paper's
# vocabulary, never re-collected — see the `collect` warning below.
# ============================================================================

SHELL := /bin/bash
ROOT := $(CURDIR)
PY ?= python3

# Experiment knobs, threaded to the lane harnesses through the environment (the
# harnesses read them as `${X:-default}`, so an unset knob keeps its own
# default). GNU make gotcha: never put an inline `#` comment on a `?=` or `:=`
# line -- make preserves the whitespace before the `#` and it lands in the value.
JOBS ?=
DEVICE ?=
RTC ?=
N_P1 ?=
N_CELLS ?=
LEVEL2 ?=
ONLY ?=
LIMIT ?=
export JOBS
export DEVICE
export RTC
export N_P1
export N_CELLS
export LEVEL2
export ONLY
export LIMIT

# Ready lanes and their stats.json (NOTE: triton writes under triton/results/,
# the other two under results/<lane>/ — render.py normalizes this).
LANES := iree triton choreo

RENDER_DIR ?= $(ROOT)/render

# Stats the renderer reads (probed by render.py; missing lanes are reported,
# not fatal): results/{iree,choreo}/stats.json and triton/results/stats.json.

# Paper workspace (sibling of svn-artifacts). Only used by `make paper`.
#
# The live venue directory is `eurosys27` (the paper was retargeted from ASPLOS
# to EuroSys; `asplos27` survives only as the pre-retarget snapshot). Override
# with `make paper PAPER=/path/to/venue` if the directory is named differently.
PAPER ?= $(ROOT)/../../eurosys27

# ---------------------------------------------------------------------------
# Per-lane stage passthrough targets: `make <lane>-<stage>` -> bash <lane>/run.sh <stage>
# ---------------------------------------------------------------------------
define LANE_STAGE
.PHONY: $(1)-$(2)
$(1)-$(2):
	bash $(ROOT)/$(1)/run.sh $(2)
endef

$(foreach s,setup minimal e2 e3 collect stats all,$(eval $(call LANE_STAGE,iree,$(s))))
$(foreach s,setup minimal e2 e3 collect stats all,$(eval $(call LANE_STAGE,triton,$(s))))
$(foreach s,setup screen minimal e2 e3 e4 e5 e5b never collect stats all,$(eval $(call LANE_STAGE,choreo,$(s))))

# ---------------------------------------------------------------------------
# GATE 2 screen — registry integrity + corpus census, writes NO mutant.
#
# Exit codes from gen_mutants.py are the gate: 0 clean, 1 a mutation edit that
# did not match (a malformed operator silently yields ZERO mutants, and only the
# `skipped` count reveals it), 2 open work (an ADMISSIBLE spec with no operator)
# or an unresolvable spec. Run this before generating a corpus.
# ---------------------------------------------------------------------------
.PHONY: screen
screen: choreo-screen

# ---------------------------------------------------------------------------
# Guard integrity -- the axis checker and its control harness.
#
# `guards` runs the checker; `test-guards` mutates one input at a time and
# requires each guard to refuse it, restoring byte-identically in between. The
# controls are the reason to trust a green `guards`: a guard that cannot fail
# and a guard that passes look the same, so `test-guards` asserts that every id
# in `check_class_axis.py --list` has a control or a stated reason to lack one.
# Run both; `guards` alone cannot tell you it is still checking anything.
# ---------------------------------------------------------------------------
.PHONY: guards test-guards
guards:
	$(PY) $(ROOT)/schema/check_class_axis.py

test-guards:
	$(PY) $(ROOT)/schema/test_guards.py

# ---------------------------------------------------------------------------
# Dashboard -- the progress view.
#
# Read-only projection of what is on disk: families x classes x lanes with
# planned/instance counts, declaration health (`R_f`/`R_a`), the tracked
# blockers, and the dated progress log from `schema/dashboard-log.md`.
#
# It NEVER writes into a corpus, manifest or results file, so it is safe to run
# at any time, including mid-generation. Regenerate it after every change that
# moves a number, and add an entry to `schema/dashboard-log.md` so the history
# survives.
#
# Output defaults to the LIVE venue directory ($(PAPER)/DASHBOARD.md, i.e.
# ../eurosys27/). `asplos27` is the pre-retarget snapshot and must not be
# written to. Override with `make dashboard OUT=/some/path.md`.
# ---------------------------------------------------------------------------
.PHONY: dashboard
OUT ?= $(PAPER)/DASHBOARD.md
OUT_WORKLIST ?= $(PAPER)/plan/m1-m4-handoff/worklist.csv
OUT_FILL ?= $(PAPER)/plan/m1-m4-handoff/FILL-DASHBOARD.md
dashboard:
	$(PY) $(ROOT)/schema/gen_dashboard.py $(OUT)

# ---------------------------------------------------------------------------
# Fill dashboard -- planned vs implemented, per (lane, class, family).
#
# `dashboard` answers "which classes are behind". This answers the next
# question down -- which family, in which lane, is short and by how much --
# which is what decides whether the next hour goes into an operator or into a
# kernel. It reads `worklist.csv` and the taxonomy's scope model, so it cannot
# disagree with `dashboard` about a lane's plan.
#
# Read-only. Regenerate after any corpus change; it is a projection, never a
# hand-maintained table. Override with `make fill-dashboard
# OUT_FILL=/some/path.md`.
# ---------------------------------------------------------------------------
.PHONY: fill-dashboard
fill-dashboard:
	$(PY) $(ROOT)/schema/gen_fill_dashboard.py $(OUT_WORKLIST) $(OUT_FILL)

# ---------------------------------------------------------------------------
# Worklist -- the per-family assignment sheet for a class handoff.
#
# One row per (lane, class, family) with the instance gap, a `fill_plan` that
# opens kernels rather than padding counts (because N = 4 kernels x 2
# realisations, so a family is never a flat count of eight), and a `blocker`
# naming the spec-level reason a family cannot move.
#
# The `state` column is the one to read first:
#   in-scope                  assignable work
#   UNCOMPARED-conflict       in the taxonomy's scope but `uncompared` on the
#                             class axis -- a conflict between two instruments,
#                             NOT a gap. Resolve before assigning.
#   unattributed(no spec_id)  the lane's rows carry no spec_id, so no family can
#                             be assigned. Fix the collector, not the corpus.
#   carved-out(<why>)         a stated lane_scope exclusion
#   out-of-scope              the lane does not run this class
#
# Read-only, same guarantees as `dashboard`. Output defaults into the LIVE
# venue directory, next to the docs the handoff cites
# ($(PAPER)/plan/m1-m4-handoff/worklist.csv). Override with
# `make worklist OUT=/some/path.csv`. Scope is set in the generator
# (`CLASSES`, `LANES`) so the same script emits the M2/M3 handoff later.
# ---------------------------------------------------------------------------
.PHONY: worklist
worklist:
	$(PY) $(ROOT)/schema/gen_worklist.py $(OUT_WORKLIST)

# ---------------------------------------------------------------------------
# Lane-level: full pipeline
# ---------------------------------------------------------------------------
.PHONY: $(LANES)
iree:   iree-all
triton: triton-all
choreo: choreo-all

# ---------------------------------------------------------------------------
# Experiments (aggregate across ready lanes)
# ---------------------------------------------------------------------------
.PHONY: experiments e1 e2 e3 e4 e5 collect stats

# Full pipelines, choreo last (it owns the exclusive-GPU timing lanes).
# `screen` is an order-only prerequisite: the corpus gate runs first and a
# failure stops the sweep, so a malformed operator can never burn GPU hours.
experiments: | screen
experiments: iree-all triton-all choreo-all

# One experiment stage across every lane that supports it.
e1: | screen
e1: iree-minimal triton-minimal choreo-minimal
e2: iree-e2 triton-e2 choreo-e2
e3: iree-e3 triton-e3 choreo-e3
e4: choreo-e4          # compile cost — choreo only (S10)
e5: choreo-e5          # runtime residue + latency — choreo only (S13)

# collect + stats (re-aggregation of raw/ -> results/*).
#
# WARNING: `collect` is NOT idempotent across the v2.1 boundary. It reads
# choreo/raw/e1_mutant_records.json and REFUSES a manifest whose mutant rows
# carry no `spec_id` (collect.py raises), because emitting a partially
# path-classified register would make S14 silently partial. The committed E1
# baseline predates v2.1, so on a clean checkout `make choreo-collect` fails by
# design and the frozen results/ stay as they are. To re-collect, first run
# `make choreo-minimal`, which regenerates the records from the v2.1 manifest.
collect: iree-collect triton-collect choreo-collect
stats:   iree-stats   triton-stats   choreo-stats

# ---------------------------------------------------------------------------
# Render (merge stats.json -> summary + tables + figures)
# ---------------------------------------------------------------------------
.PHONY: render summary tables figures drift

summary:
	$(PY) $(ROOT)/render.py --summary --out $(RENDER_DIR)

tables:
	$(PY) $(ROOT)/render.py --tables --out $(RENDER_DIR)

figures:
	$(PY) $(ROOT)/render.py --figures --out $(RENDER_DIR)

# Re-check every numeric claim in the paper's prose against the committed
# statistics and rewrite render/PROSE-DRIFT.md. Needs the paper workspace.
drift:
	$(PY) $(ROOT)/render.py --drift --paper-dir $(PAPER) --out $(RENDER_DIR)

render: summary tables figures drift

# ---------------------------------------------------------------------------
# Paper sync (opt-in): copy render output into the paper workspace.
#
# Only the floats main.tex actually \input{}s are copied, and only after a
# diff. render.py also emits six reference-only floats (e1_*, rq1/rq3/rq5/rq6)
# and three generated PDFs; those stay here, otherwise the venue accumulates
# files nothing reads -- and figures/fig_e1_detection.tex would collide with
# tables/e1_detection.tex over the tab:e1-detection label.
#
# The sync refuses to overwrite a venue float with uncommitted changes, so the
# paper's own edits stay separable with `git checkout`.
# ---------------------------------------------------------------------------
PAPER_SYNC := $(ROOT)/sync_paper.sh

.PHONY: paper paper-check

# Diff-only: shows NEW / SAME / CHANGED per float and copies nothing.
paper-check: render
	@bash $(PAPER_SYNC) $(PAPER) --dry-run

paper: render
	@bash $(PAPER_SYNC) $(PAPER)

# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------
.PHONY: all clean
all: experiments stats render

clean:
	rm -rf $(RENDER_DIR)
	@echo "[clean] removed $(RENDER_DIR) (lane results left untouched)"

help:
	@sed -n '/^# ==========/,/^# ==========/p' $(ROOT)/Makefile | sed 's/^# \{0,1\}//'
