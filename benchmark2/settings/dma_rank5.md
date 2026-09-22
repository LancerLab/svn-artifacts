# dma_rank5 — operator settings

- **operator**: `dma_rank5`
- **reference semantics**: rank-5 dma.transp descriptor surface used only by the M3 mutation specs (r5dyn: global->shared with a symbolic leading dim; r5base: a well-formed rank-5 permutation). Not a benchmark operator.
- **cases**: 2 (derived from `benchmark/choreo/dma_rank5/`; signatures only, not source)

## Cases (provenance)

| # | source file | content-sha1 | operator signature |
|---|---|---|---|
| 1 | `benchmark/choreo/dma_rank5/r5base.co` | `71275da4ffb1` | `__co__ auto r5base(f32 [2,4,8,16,32] input) {` |
| 2 | `benchmark/choreo/dma_rank5/r5dyn.co` | `5b65ecb6c656` | `__co__ auto r5dyn(f32 [N,2,1,1,1] input) {` |
