# matmul — operator settings

- **operator**: `matmul`
- **reference semantics**: y = lhs @ rhs; contract the inner dim K. Output = lhs.shape[:-1] + rhs.shape[-1:].
- **cases**: 20 (derived from `benchmark/choreo/matmul/`; signatures only, not source)

## Cases (provenance)

| # | source file | content-sha1 | operator signature |
|---|---|---|---|
| 1 | `benchmark/choreo/matmul/10_dynamic_128x1280_1280xN_128xN.co` | `bc33893a6e95` | `__co__ auto matmul10(f32 [128, 1280] lhs, f32 [1280, N] rhs) {` |
| 2 | `benchmark/choreo/matmul/11_dynamic_32xSx768_768x768_32xSx768.co` | `c8f414c3fb0f` | `__co__ auto matmul11(f32 [32, S, 768] lhs, f32 [768, 768] rhs) {` |
| 3 | `benchmark/choreo/matmul/12_dynamic_64xTx256_256x128_64xTx128.co` | `be82ea044761` | `__co__ auto matmul12(f32 [64, T, 256] lhs, f32 [256, 128] rhs) {` |
| 4 | `benchmark/choreo/matmul/13_dynamic_32x512xV_Vx768_32x512x768.co` | `563d084bdd42` | `__co__ auto matmul13(f32 [32, 512, V] lhs, f32 [V, 768] rhs) {` |
| 5 | `benchmark/choreo/matmul/14_efficientnet_64x1280_1280x1000_64x1000.co` | `c50f41601a3a` | `__co__ auto matmul14(f32 [64, 1280] lhs, f32 [1280, 1000] rhs) {` |
| 6 | `benchmark/choreo/matmul/15_gpt_16x1024x1536_1536x6144_16x1024x6144.co` | `ef156085e73c` | `__co__ auto matmul15(f32 [16, 1024, 1536] lhs, f32 [1536, 6144] rhs) {` |
| 7 | `benchmark/choreo/matmul/16_lstm_64x100x300_300x256_64x100x256.co` | `ee14ebe020c3` | `__co__ auto matmul16(f32 [64, 100, 300] lhs, f32 [300, 256] rhs) {` |
| 8 | `benchmark/choreo/matmul/17_general_32x512x512_512x512_32x512x512.co` | `c935dea9d168` | `__co__ auto matmul17(f32 [32, 512, 512] lhs, f32 [512, 512] rhs) {` |
| 9 | `benchmark/choreo/matmul/18_resnet_128x2048_2048x1000_128x1000.co` | `f17b3ebc7064` | `__co__ auto matmul18(f32 [128, 2048] lhs, f32 [2048, 1000] rhs) {` |
| 10 | `benchmark/choreo/matmul/19_transformer_64x512x512_512x2048_64x512x2048.co` | `a063481b4a5e` | `__co__ auto matmul19(f32 [64, 512, 512] lhs, f32 [512, 2048] rhs) {` |
| 11 | `benchmark/choreo/matmul/1_bert_32x512x768_768x768_32x512x768.co` | `fa0cbf2517c6` | `__co__ auto matmul1(f32 [32, 512, 768] lhs, f32 [768, 768] rhs) {` |
| 12 | `benchmark/choreo/matmul/20_vit_32x197x768_768x3072_32x197x3072.co` | `2f1115c1a648` | `__co__ auto matmul20(f32 [32, 197, 768] lhs, f32 [768, 3072] rhs) {` |
| 13 | `benchmark/choreo/matmul/2_bert_32x512x30522_30522x768_32x512x768.co` | `42118125c99f` | `__co__ auto matmul2(f32 [32, 512, 762] lhs, f32 [762, 768] rhs)  {` |
| 14 | `benchmark/choreo/matmul/3_cnn_256x512_512x10_256x10.co` | `24fcc3c5e605` | `__co__ auto matmul3(f32 [256, 512] lhs, f32 [512, 10] rhs) {` |
| 15 | `benchmark/choreo/matmul/4_dynamic_32xNx512x64_32xNx64x512_32xNx512x512.co` | `358f336da1ad` | `__co__ auto matmul4(f32 [32, num_heads, 512, 64] lhs, f32 [32, num_heads, 64, 512] rhs) {` |
| 16 | `benchmark/choreo/matmul/5_dynamic_Bx2048_2048x1000_Bx1000.co` | `c70897dcfbd8` | `__co__ auto matmul5(f32 [batch_size, 2048] lhs, f32 [2048, 1000] rhs) {` |
| 17 | `benchmark/choreo/matmul/6_dynamic_32x197xE_Ex3072_32x197x3072.co` | `bac84878856e` | `__co__ auto matmul6(f32 [32, 197, E] lhs, f32 [E, 3072] rhs) {` |
| 18 | `benchmark/choreo/matmul/7_dynamic_16x1024xD_Dx4096_16x1024x4096.co` | `6bbd98ddbd4d` | `__co__ auto matmul7(f32 [16, 1024, D] lhs, f32 [D, 4096] rhs) {` |
| 19 | `benchmark/choreo/matmul/8_dynamic_64x100x300_300xH_64x100xH.co` | `6b5cd5d309b1` | `__co__ auto matmul8(f32 [64, 100, 300] lhs, f32 [300, H] rhs) {` |
| 20 | `benchmark/choreo/matmul/9_dynamic_16xSx512_512x256_16xSx256.co` | `f10442028961` | `__co__ auto matmul9(f32 [16, S, 512] lhs, f32 [512, 256] rhs) {` |
