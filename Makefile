SHELL := /bin/bash

ROOT := $(CURDIR)
CHOREO_DIR := $(ROOT)/choreo
TRITON_VENV := $(ROOT)/external/triton-v3.6.0/.venv
TRITON_PYTHON := $(TRITON_VENV)/bin/python

.PHONY: help reproduce-all choreo-build choreo-test choreo-stats choreo-cto \
       bug-detect mlir-clone mlir-build compare compare-triton \
       generate-triton-suite render-triton-cases \
       test-triton-unary-chunks test-triton-binary-chunks \
       test-triton test-triton-correctness plot clean-results

help:
	@echo "SVN Artifact Evaluation (CGO 2027)"
	@echo "==================================="
	@echo ""
	@echo "Quick start:"
	@echo "  make reproduce-all            - One-command full evaluation"
	@echo ""
	@echo "Individual steps:"
	@echo "  make choreo-build             - Build Choreo compiler from source"
	@echo "  make choreo-test              - Run compile-time tests (check + cli)"
	@echo "  make choreo-stats             - RQ1: collect assessment statistics"
	@echo "  make bug-detect               - RQ2: bug detection evaluation"
	@echo "  make choreo-cto               - RQ4: measure compile-time overhead"
	@echo ""
	@echo "MLIR baseline (optional, ~30 min build):"
	@echo "  make mlir-clone               - Clone llvm-project release/22.x"
	@echo "  make mlir-build               - Build mlir-opt, mlir-translate, FileCheck"
	@echo "  make compare                  - Run SVN vs MLIR comparison"
	@echo ""
	@echo "Triton baseline (optional):"
	@echo "  make compare-triton           - Run SVN vs Triton comparison"
	@echo "  make test-triton              - All Triton cases (compile-only)"
	@echo ""
	@echo "Utilities:"
	@echo "  make plot                     - Regenerate figures"
	@echo "  make clean-results            - Remove generated benchmark outputs"

reproduce-all:
	bash scripts/reproduce_all.sh

choreo-build:
	$(MAKE) -C $(CHOREO_DIR) release

choreo-test:
	cd $(CHOREO_DIR) && bash tests/lit.sh tests/check && bash tests/lit.sh tests/cli

choreo-stats:
	python3 scripts/choreo_assertion_stats.py --choreo $(CHOREO_DIR)/choreo

bug-detect:
	python3 scripts/bug_detection_eval.py

choreo-cto:
	python3 scripts/choreo_compile_overhead.py --reps 3

mlir-clone:
	bash scripts/clone_llvm_mlir.sh

mlir-build:
	bash scripts/build_llvm_mlir.sh

compare:
	bash scripts/compare_choreo_mlir.sh

compare-triton:
	bash scripts/compare_choreo_triton.sh

generate-triton-suite:
	$(TRITON_PYTHON) scripts/generate_triton_benchmark_suite.py

render-triton-cases:
	$(TRITON_PYTHON) scripts/render_triton_standalone_cases.py

test-triton-unary-chunks:
	$(TRITON_PYTHON) benchmark/triton/tests/test_unary_chunk_data.py --device cuda

test-triton-binary-chunks:
	$(TRITON_PYTHON) benchmark/triton/tests/test_binary_chunk_data.py --device cuda

test-triton:
	$(TRITON_PYTHON) benchmark/triton/tests/test_all_cases.py --mode compile-only

test-triton-correctness:
	$(TRITON_PYTHON) benchmark/triton/tests/test_all_cases.py --mode chunk-check

plot:
	bash scripts/render_paper_figures.sh

clean-results:
	rm -rf benchmark/results benchmark/mlir/results benchmark/triton/results
