# embedding — operator settings

- **operator**: `embedding`
- **reference semantics**: y = w[id]; gather rows of w (vocab_size x embed_dim) indexed by integer ids. Output = id.shape + [embed_dim].
- **cases**: 20 (derived from `benchmark/choreo/embedding/`; signatures only, not source)

## Cases (provenance)

| # | source file | content-sha1 | operator signature |
|---|---|---|---|
| 1 | `benchmark/choreo/embedding/10_dynamic_32xS_Vx768_32xSx768.co` | `2be858543c6b` | `__co__ auto EMBEDDING(s32 [32, seq_len] id, f32 [vocab_size, 768] w) {` |
| 2 | `benchmark/choreo/embedding/11_dynamic_64xN_40000x512_64xNx512.co` | `ca42351132db` | `__co__ auto EMBEDDING(s32 [64, num_subwords] id, f32 [40000, 512] w) {` |
| 3 | `benchmark/choreo/embedding/12_dynamic_64x256_VxD_64x256xD.co` | `fd4a089c7d17` | `__co__ auto EMBEDDING(s32 [64, 256] id, f32 [vocab_size, embed_dim] w) {` |
| 4 | `benchmark/choreo/embedding/13_dynamic_NxW_30522x768_NxWx768.co` | `c19aee21a832` | `__co__ auto EMBEDDING(s32 [batch_size, num_wordpieces] id, f32 [30522, 768] w) {` |
| 5 | `benchmark/choreo/embedding/14_gpt_16x1024_50257x1024_16x1024x1024.co` | `364bae5ec91c` | `__co__ auto EMBEDDING(s32 [16, 1024] id, f32 [50257, 1024] w) {` |
| 6 | `benchmark/choreo/embedding/15_bert_32x512_30522x768_32x512x768.co` | `26c42b7d7d8c` | `__co__ auto EMBEDDING(s32 [32, 512] id, f32 [30522, 768] w) {` |
| 7 | `benchmark/choreo/embedding/16_general_64x256_20000x512_64x256x512.co` | `4ea439ed416f` | `__co__ auto EMBEDDING(s32 [64, 256] id, f32 [20000, 512] w) {` |
| 8 | `benchmark/choreo/embedding/17_general_16x512_25000x768_16x512x768.co` | `90e2b19664df` | `__co__ auto EMBEDDING(s32 [16, 512] id, f32 [25000, 768] w) {` |
| 9 | `benchmark/choreo/embedding/18_transformer_32x512_32000x512_32x512x512.co` | `48ee3064892b` | `__co__ auto EMBEDDING(s32 [32, 512] id, f32 [32000, 512] w) {` |
| 10 | `benchmark/choreo/embedding/19_vit_32x197_1000x768_32x197x768.co` | `2751e5e740d3` | `__co__ auto EMBEDDING(s32 [32, 197] id, f32 [1000, 768] w) {` |
| 11 | `benchmark/choreo/embedding/1_bert_32x512_30522x768_32x512x768.co` | `26c42b7d7d8c` | `__co__ auto EMBEDDING(s32 [32, 512] id, f32 [30522, 768] w) {` |
| 12 | `benchmark/choreo/embedding/20_general_64x100_10000x300_64x100x300.co` | `5ead2e2e3097` | `__co__ auto EMBEDDING(s32 [64, 100] id, f32 [10000, 300] w) {` |
| 13 | `benchmark/choreo/embedding/2_general_128x50_10000x128_128x50x128.co` | `8f736589a36b` | `__co__ auto EMBEDDING(s32 [128, 50] id, f32 [10000, 128] w) {` |
| 14 | `benchmark/choreo/embedding/3_cnn_128x1000_50000x256_128x1000x256.co` | `d0d9aed99d89` | `__co__ auto EMBEDDING(s32 [128, 1000] id, f32 [50000, 256] w) {` |
| 15 | `benchmark/choreo/embedding/4_dynamic_Nx512_Vx768_Nx512x768.co` | `13d8fafbd322` | `__co__ auto EMBEDDING(s32 [batch_size, 512] id, f32 [vocab_size, 768] w) {` |
| 16 | `benchmark/choreo/embedding/5_dynamic_32xN_50000x1024_32xNx1024.co` | `7af9d60f1b87` | `__co__ auto EMBEDDING(s32 [32, num_bpe_tokens] id, f32 [50000, 1024] w) {` |
| 17 | `benchmark/choreo/embedding/6_dynamic_16xN_256x128_16xNx128.co` | `0c9b1ef62b13` | `__co__ auto EMBEDDING(s32 [16, num_chars] id, f32 [256, 128] w) {` |
| 18 | `benchmark/choreo/embedding/7_dynamic_128xS_30522x768_128xSx768.co` | `aa44d77ac91d` | `__co__ auto EMBEDDING(s32 [128, seq_len] id, f32 [30522, 768] w) {` |
| 19 | `benchmark/choreo/embedding/8_dynamic_16xN_512x768_128xSx768.co` | `bb2642c6182a` | `__co__ auto EMBEDDING(s32 [16, num_positions] id, f32 [512, 768] w) {` |
| 20 | `benchmark/choreo/embedding/9_dynamic_32xN_32000x768_32xNx768.co` | `937481b9895c` | `__co__ auto EMBEDDING(s32 [32, num_pieces] id, f32 [32000, 768] w) {` |
