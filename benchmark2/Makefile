# ============================================================================
# benchmark2 Makefile — ASPLOS 2027 artifact orchestration.
#
# Ties the per-lane `run.sh` harnesses together: run each experiment, collect
# per-lane statistics, then merge them into the paper's tables/figures.
#
#   make help           - this help
#   make all            - experiments + stats + render (the full pipeline)
#   make experiments    - run every ready lane's `run.sh all`
#   make e1|e2|e3       - run one experiment stage across the ready lanes
#   make e4|e5          - compile-cost / runtime-residue (choreo only, exclusive)
#   make iree|triton|choreo  - run a single lane's full pipeline
#   make <lane>-<stage> - run one stage of one lane (e.g. make triton-stats)
#   make stats          - re-run collect+stats for every ready lane
#   make render         - merge stats.json -> render/ (summary + tables + figures)
#   make tables|figures - only the tables / figures
#   make paper          - copy render output into the paper workspace (../asplos27)
#   make clean          - remove render/ output (lane results are NOT touched)
#
# Ready lanes: iree, triton, choreo.
# NOT ready yet (excluded on purpose): mlir-linalg, mlir-low, tilelang — they
# have no results/ outputs yet (see homework-check.md "Out of scope"). When they
# push, add them to LANES below and to render.py's LANE_STATS_PATHS.
# ============================================================================

SHELL := /bin/bash
ROOT := $(CURDIR)
PY ?= python3

# Ready lanes and their stats.json (NOTE: triton writes under triton/results/,
# the other two under results/<lane>/ — render.py normalizes this).
LANES := iree triton choreo

RENDER_DIR ?= $(ROOT)/render
TABLES_DIR := $(RENDER_DIR)/tables
FIGURES_DIR := $(RENDER_DIR)/figures

# Stats the renderer reads (probed by render.py; missing lanes are reported,
# not fatal): results/{iree,choreo}/stats.json and triton/results/stats.json.

# Paper workspace (sibling of svn-artifacts). Only used by `make paper`.
PAPER ?= $(ROOT)/../../asplos27

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
$(foreach s,setup minimal e2 e3 e4 e5 e5b never collect stats all,$(eval $(call LANE_STAGE,choreo,$(s))))

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
experiments: iree-all triton-all choreo-all

# One experiment stage across every lane that supports it.
e1: iree-minimal triton-minimal choreo-minimal
e2: iree-e2 triton-e2 choreo-e2
e3: iree-e3 triton-e3 choreo-e3
e4: choreo-e4          # compile cost — choreo only (S10)
e5: choreo-e5          # runtime residue + latency — choreo only (S13)

# collect + stats (idempotent re-aggregation of raw/ -> results/*).
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
