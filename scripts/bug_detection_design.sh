#!/bin/bash
# Bug Detection Comparison: Choreo vs Triton vs MLIR vs IREE
# ============================================================
#
# EXPERIMENT DESIGN (based on literature-grounded bug taxonomy)
#
# References:
#   [1] SCuBA: "Chasing Elusive Memory Bugs in GPU Programs" (2026)
#       - Input-dependent OOB, Intra-allocation OOB
#   [2] Tensfa/SFData: "Empirical Study on Tensor Shape Faults" (ISSRE 2021)
#       - 4 types: incompatible shapes, wrong transformation, missing dim, inference failure
#   [3] GPU-Fuzz: "Finding Memory Errors in Deep Learning Frameworks" (2026)
#       - Operator-level memory errors from tensor shapes/strides/parameters
#   [4] TSE 2022: "DL Framework Bug Study" (1000 bugs across TF/PyTorch/MXNet/DL4J)
#       - Tensor Shape Misalignment causes crashes AND silent incorrect functionality
#   [5] ShapeTracer (Alibaba): 12,289 failed TF jobs; Shape Errors = 8.82% of all TF bugs
#   [6] Wu et al. (2019): "Characterizing and Detecting CUDA Program Bugs"
#
# METHODOLOGY
# ===========
# We inject 5 classes of bugs (from the taxonomy) into our 310-case benchmark
# suite and compare each system's ability to detect them:
#
# Bug Class 1: DIMENSION MISMATCH (Tensfa Type 1, TSE'22 "Shape Misalignment")
#   - Inject: Pass gamma with dim V-1 instead of V (off-by-one dimension)
#   - Real-world analog: Model layer output dim ≠ next layer input dim
#   - Detection mechanism:
#     * Choreo: cross-tensor assertion gamma.span(0)==inp.span(2) → catches at ENTRY
#     * MLIR: cf.assert gamma.dim(0)==input.dim(2) → catches (MANUAL assertion)
#     * IREE: entry-boundary shape check → catches (shape literally wrong)
#     * Triton: require() if developer wrote shape check → depends on coverage
#
# Bug Class 2: INPUT-DEPENDENT OOB (SCuBA Category)
#   - Inject: Set dynamic dim to value that causes tile overflow
#             (e.g., V=255 when tile_size=256, causing last tile to OOB)
#   - Real-world analog: Batch sizes/sequence lengths that don't divide evenly
#   - Detection mechanism:
#     * Choreo: runtime assertion on .at() bounds → HOIST or USE_SITE catches
#     * MLIR: tensor.extract has automatic bounds check (if verification pass on)
#     * IREE: no interior check → silent OOB or CUDA error 700
#     * Triton: tl.load with mask may silently read garbage; no auto-check
#
# Bug Class 3: INTRA-ALLOCATION OOB (SCuBA Category, NEW)
#   - Inject: Wrong offset calculation in shared/local memory partitioning
#             (e.g., two buffers sharing smem, one overflows into other)
#   - Real-world analog: Shared memory buffer overflow in tiled kernels
#   - Detection mechanism:
#     * Choreo: .at() on local/shared tensor catches bounds violation
#     * MLIR: no check (scf loops don't validate logical boundaries)
#     * IREE: no check (operates at function boundary only)
#     * Triton: no check (tl.store to shared memory unchecked)
#
# Bug Class 4: STRIDE/LAYOUT ERROR (GPU-Fuzz, TSE'22)
#   - Inject: Wrong stride in reshape/view (e.g., interpret [N,S,E] as [N,E,S])
#   - Real-world analog: Transposing dimensions without updating access pattern
#   - Detection mechanism:
#     * Choreo: shape assertion detects dimension mismatch on operation
#     * MLIR: no automatic check (layout encoded implicitly in loop bounds)
#     * IREE: no check (layout is part of encoding, not verified at interior)
#     * Triton: no check (manual pointer arithmetic)
#
# Bug Class 5: ARITHMETIC OVERFLOW IN INDEX (GPU Taxonomy Class 16)
#   - Inject: Integer overflow in index computation for large tensors
#             (e.g., N*S*E > INT_MAX for 32-bit index)
#   - Real-world analog: Large batch × long sequence overflows 32-bit indexing
#   - Detection mechanism:
#     * Choreo: bounds assertion fires when computed index exceeds span
#     * MLIR: no automatic overflow detection
#     * IREE: no check
#     * Triton: silent overflow → wrong memory access
#
# IMPLEMENTATION PLAN
# ===================
# For each of the 5 bug classes, we:
#   1. Select 10 representative cases (2 from each major category: batch_norm,
#      layer_norm, softmax, matmul, elemwise)
#   2. Create a mutant for each case with the specific bug injected
#   3. Run each system with maximum assertion coverage
#   4. Record: {detected_compile_time, detected_runtime, undetected, crash_opaque}
#
# Total: 5 bug_classes × 10 cases = 50 mutants per system × 4 systems = 200 runs
#
# WHAT WE MEASURE
# ===============
# For each (bug_class, case, system) triple:
#   - outcome ∈ {STATIC_DETECT, RUNTIME_DETECT, CRASH_OPAQUE, SILENT_WRONG, N/A}
#     * STATIC_DETECT: compile-time error/assertion that identifies the bug
#     * RUNTIME_DETECT: runtime assertion with clear message identifying the bug
#     * CRASH_OPAQUE: program crashes but error message is unhelpful (e.g., "CUDA error 700")
#     * SILENT_WRONG: program runs but produces incorrect results silently
#     * N/A: system doesn't support this case (e.g., MLIR memref for dynamic shapes)
#
# EXPECTED RESULTS (hypothesis)
# ==============================
# | Bug Class           | Choreo     | MLIR (manual) | IREE     | Triton   |
# |---------------------|------------|---------------|----------|----------|
# | Dim mismatch        | STATIC 90% | RUNTIME 80%   | RUNTIME  | SILENT   |
# | Input-dep OOB       | RUNTIME    | RUNTIME 50%   | CRASH    | SILENT   |
# | Intra-alloc OOB     | RUNTIME    | SILENT        | CRASH    | SILENT   |
# | Stride/layout error | STATIC     | SILENT        | SILENT   | SILENT   |
# | Index overflow      | RUNTIME    | SILENT        | SILENT   | SILENT   |
#
# KEY NARRATIVE FOR PAPER
# ========================
# 1. Choreo achieves highest detection rate with ZERO manual effort
# 2. MLIR can match for dim-mismatch but ONLY with 272 manually added assertions
# 3. IREE catches entry-level mismatches but misses all interior bugs
# 4. Triton delegates everything to the developer — misses any un-asserted bug
# 5. The advantage is structural: semantic-stage analysis sees contracts that
#    are invisible after lowering (SCuBA's key insight applies at the IR level)

echo "This is a design document. See scripts/bug_detection_run.sh for execution."
