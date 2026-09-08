# batch_norm — operator settings

- **operator**: `batch_norm`
- **reference semantics**: y[n,c,...] = (x[n,c,...] - mean_c)/sqrt(var_c+eps) * gamma[c] + beta[c]; mean/var reduced over the non-channel dims (training mode, eps=1e-5).
- **cases**: 21 (derived from `benchmark/choreo/batch_norm/`; signatures only, not source)

## Cases (provenance)

| # | source file | content-sha1 | operator signature |
|---|---|---|---|
| 1 | `benchmark/choreo/batch_norm/10_dynamic_16x512xHxW_512_512_16x512xHxW.co` | `9a93c7dbc6fe` | `__co__ f32 [N, C, H, W] ele_batch_norm(f32 [N, C, H, W] inp, f32 [C] gamma, f32 [C] beta) {` |
| 2 | `benchmark/choreo/batch_norm/11_dynamic_32xSx768_768_768_32xSx768.co` | `aee16b19609a` | `__co__ f32 [N, S, E] ele_batch_norm(f32 [N, S, E] inp, f32 [E] gamma, f32 [E] beta) {` |
| 3 | `benchmark/choreo/batch_norm/12_dynamic_64xTx256_256_256_64xTx256.co` | `d948634efd6c` | `__co__ f32 [N, S, E] ele_batch_norm(f32 [N, S, E] inp, f32 [E] gamma, f32 [E] beta) {` |
| 4 | `benchmark/choreo/batch_norm/13_dynamic_32x512xV_V_V_32x512xV.co` | `59bca5027df1` | `__co__ f32 [N, S, E] ele_batch_norm(f32 [N, S, E] inp, f32 [E] gamma, f32 [E] beta) {` |
| 5 | `benchmark/choreo/batch_norm/14_efficientnet_64x1280x7x7_1280_1280_64x1280x7x7.co` | `d0328352a2bc` | `__co__ f32 [N, C, H, W] ele_batch_norm(f32 [N, C, H, W] inp, f32 [C] gamma, f32 [C] beta) {` |
| 6 | `benchmark/choreo/batch_norm/15_gpt_16x1024x4096_4096_4096_16x1024x4096.co` | `986491ce872d` | `__co__ f32 [N, S, E] ele_batch_norm(f32 [N, S, E] inp, f32 [E] gamma, f32 [E] beta) {` |
| 7 | `benchmark/choreo/batch_norm/16_lstm_64x100x256_256_256_64x100x256.co` | `219334a868c1` | `__co__ f32 [N, S, E] ele_batch_norm(f32 [N, S, E] inp, f32 [E] gamma, f32 [E] beta) {` |
| 8 | `benchmark/choreo/batch_norm/17_mobilenet_128x96x112x112_96_96_128x96x112x112.co` | `b6be64ded649` | `__co__ f32 [N, C, H, W] ele_batch_norm(f32 [N, C, H, W] inp, f32 [C] gamma, f32 [C] beta) {` |
| 9 | `benchmark/choreo/batch_norm/18_resnet_64x256x56x56_256_256_64x256x56x56.co` | `9fe1a7a561b7` | `__co__ f32 [N, C, H, W] ele_batch_norm(f32 [N, C, H, W] inp, f32 [C] gamma, f32 [C] beta) {` |
| 10 | `benchmark/choreo/batch_norm/19_transformer_32x512x2048_2048_2048_32x512x2048.co` | `a98f386131a1` | `__co__ f32 [N, S, E] ele_batch_norm(f32 [N, S, E] inp, f32 [E] gamma, f32 [E] beta) {` |
| 11 | `benchmark/choreo/batch_norm/1_bert_32x512x768_512_512_32x512x768.co` | `45b43c2f6577` | `__co__ f32 [N, S, H] ele_batch_norm(f32 [N, S, H] inp, f32 [S] gamma, f32 [S] beta) {` |
| 12 | `benchmark/choreo/batch_norm/20_unet_16x512x32x32_512_512_16x512x32x32.co` | `c949bd743333` | `__co__ f32 [N, C, H, W] ele_batch_norm(f32 [N, C, H, W] inp, f32 [C] gamma, f32 [C] beta) {` |
| 13 | `benchmark/choreo/batch_norm/21_vit_32x197x3072_3072_3072_32x197x3072.co` | `8037a97457ab` | `__co__ f32 [N, S, E] ele_batch_norm(f32 [N, S, E] inp, f32 [E] gamma, f32 [E] beta) {` |
| 14 | `benchmark/choreo/batch_norm/2_cnn_128x128x28x28_128_128_128x128x28x28.co` | `33bfb80db2ae` | `__co__ f32 [N, C, H, W] ele_batch_norm(f32 [N, C, H, W] inp, f32 [C] gamma, f32 [C] beta) {` |
| 15 | `benchmark/choreo/batch_norm/3_attention_32xNx512x64_N_N_32xNx512x64.co` | `1e27e9f9ebb4` | `__co__ f32 [N, C, H, W] ele_batch_norm(f32 [N, C, H, W] inp, f32 [C] gamma, f32 [C] beta) {` |
| 16 | `benchmark/choreo/batch_norm/4_dynamic_Nx256x56x56_256_256_Nx256x56x56.co` | `9a34e83d6320` | `__co__ f32 [N, C, H, W] ele_batch_norm(f32 [N, C, H, W] inp, f32 [C] gamma, f32 [C] beta) {` |
| 17 | `benchmark/choreo/batch_norm/5_dynamic_Nx1280xHxW_1280_1280_Nx1280xHxW.co` | `3aca880549cc` | `__co__ f32 [N, C, H, W] ele_batch_norm(f32 [N, C, H, W] inp, f32 [C] gamma, f32 [C] beta) {` |
| 18 | `benchmark/choreo/batch_norm/6_dynamic_128xCx112x112_C_C_128xCx112x112.co` | `f09a0e1639a8` | `__co__ f32 [N, C, H, W] ele_batch_norm(f32 [N, C, H, W] inp, f32 [C] gamma, f32 [C] beta) {` |
| 19 | `benchmark/choreo/batch_norm/7_dynamic_32x197xE_E_E_32x197xE.co` | `3699aa8d3df9` | `__co__ f32 [N, S, E] ele_batch_norm(f32 [N, S, E] inp, f32 [E] gamma, f32 [E] beta) {` |
| 20 | `benchmark/choreo/batch_norm/8_dynamic_16x1024xD_D_D_16x1024xD.co` | `f5c4b376d01b` | `__co__ f32 [N, S, E] ele_batch_norm(f32 [N, S, E] inp, f32 [E] gamma, f32 [E] beta) {` |
| 21 | `benchmark/choreo/batch_norm/9_dynamic_64x128xHxW_128_128_64x128xHxW.co` | `a14967e6bc42` | `__co__ f32 [N, C, H, W] ele_batch_norm(f32 [N, C, H, W] inp, f32 [C] gamma, f32 [C] beta) {` |
