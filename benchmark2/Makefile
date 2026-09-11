# ============================================================================
# benchmark2 Makefile — ASPLOS 2027 artifact orchestration.
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
#   make paper          - copy render output into the paper workspace (../eurosys27)
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
# NOT ready yet (excluded on purpose): mlir-linalg, mlir-low, tilelang — they
# have no results/ outputs yet (see homework-check.md "Out of scope"). When they
# push, add them to LANES below and to render.py's LANE_STATS_PATHS.
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
TABLES_DIR := $(RENDER_DIR)/tables
FIGURES_DIR := $(RENDER_DIR)/figures

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
.PHONY: render summary tables figures

summary:
	$(PY) $(ROOT)/render.py --summary --out $(RENDER_DIR)

tables:
	$(PY) $(ROOT)/render.py --tables --out $(RENDER_DIR)

figures:
	$(PY) $(ROOT)/render.py --figures --out $(RENDER_DIR)

render: summary tables figures

# ---------------------------------------------------------------------------
# Paper sync (opt-in): copy render output into the paper workspace.
# ---------------------------------------------------------------------------
.PHONY: paper
paper: render
	mkdir -p $(PAPER)/tables $(PAPER)/figures
	cp $(TABLES_DIR)/*.tex   $(PAPER)/tables/
	cp $(FIGURES_DIR)/*.pdf  $(PAPER)/figures/
	@echo "[paper] copied render output into $(PAPER)/{tables,figures}/"

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
