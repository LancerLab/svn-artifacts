# concat — operator settings

- **operator**: `concat`
- **reference semantics**: y = concat(a, b, axis); shapes agree on all axes except the concat axis, which sums.
- **cases**: 21 (derived from `benchmark/choreo/concat/`; signatures only, not source)

## Cases (provenance)

| # | source file | content-sha1 | operator signature |
|---|---|---|---|
| 1 | `benchmark/choreo/concat/10_dynamic_16x512xHxW_16x512xHxW_16x1024xHxW.co` | `bd91d2c3f228` | `__co__ f32 [I, J_OUT, K, L] ele_concat_J(f32 [I, J1, K, L] a, f32 [I, J2, K, L] b) {` |
| 2 | `benchmark/choreo/concat/11_dynamic_32xS1x768_32xS2x768_32xS1pS2x768.co` | `fd8aada00aa3` | `__co__ f32 [I, J_OUT, K, L] ele_concat_J(f32 [I, J1, K, L] a, f32 [I, J2, K, L] b) {` |
| 3 | `benchmark/choreo/concat/12_dynamic_64xTx256_64xTx256_64xTx512.co` | `06f5236a62da` | `__co__ f32 [I, J,  L_OUT] ele_concat_J(f32 [I, J,  L1] a, f32 [I, J,  L2] b) {` |
| 4 | `benchmark/choreo/concat/13_dynamic_32x512xV1_32x512xV2_32x512xV1pV2.co` | `91570da1e734` | `__co__ f32 [I, J, L_OUT] ele_concat_J(f32 [I, J, L1] a, f32 [I, J, L2] b) {` |
| 5 | `benchmark/choreo/concat/14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7.co` | `d816971a0a32` | `__co__ f32 [I, J_OUT, K, L] ele_concat_J(f32 [I, J1, K, L] a, f32 [I, J2, K, L] b) {` |
| 6 | `benchmark/choreo/concat/15_gpt_16x512x1536_16x512x1536_16x512x3072.co` | `f511bcdac1a2` | `__co__ f32 [I, J, 2*K, L] ele_concat_K(f32 [I, J, K, L] a, f32 [I, J, K, L] b) {` |
| 7 | `benchmark/choreo/concat/16_lstm_64x100x256_64x100x256_64x100x512.co` | `954fac32735b` | `__co__ f32 [I, J, 2*K, L] ele_concat_K(f32 [I, J, K, L] a, f32 [I, J, K, L] b) {` |
| 8 | `benchmark/choreo/concat/17_mobilenet_128x96x112x112_128x32x112x112_128x128x112x112.co` | `938d7d58eb40` | `__co__ f32 [I, J_OUT, K, L] ele_concat_J(f32 [I, J1, K, L] a, f32 [I, J2, K, L] b) {` |
| 9 | `benchmark/choreo/concat/18_resnet_64x256x56x56_64x256x56x56_64x512x56x56.co` | `b9210f7cce54` | `__co__ f32 [I, J_OUT, K, L] ele_concat_J(f32 [I, J1, K, L] a, f32 [I, J2, K, L] b) {` |
| 10 | `benchmark/choreo/concat/19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256.co` | `8462a0b21f71` | `__co__ f32 [I, J, L_OUT] ele_concat_J(f32 [I, J, L] a, f32 [I, J, L] b, f32 [I, J, L] c, f32 [I, J, L] d) {` |
| 11 | `benchmark/choreo/concat/1_bert_32x512x768_32x512x768_32x512x1536.co` | `6e0345e63aa4` | `__co__ f32 [I, J, 1536, L] ele_concat_K(f32 [I, J, K, L] a, f32 [I, J, K, L] b) {` |
| 12 | `benchmark/choreo/concat/20_unet_16x512x32x32_16x512x32x32_16x1024x32x32.co` | `2e9b4c4ca739` | `__co__ f32 [I, J_OUT, K, L] ele_concat_J(f32 [I, J1, K, L] a, f32 [I, J2, K, L] b) {` |
| 13 | `benchmark/choreo/concat/21_vit_32x196x768_32x1x768_32x197x768.co` | `f511e335dff1` | `__co__ f32 [I, J_OUT, K, L] ele_concat_J(f32 [I, J1, K, L] a, f32 [I, J2, K, L] b) {` |
| 14 | `benchmark/choreo/concat/2_cnn_128x128x28x28_128x256x28x28_128x384x28x28.co` | `fd6e8a13455e` | `__co__ f32 [I, J_OUT, K, L] ele_concat_J(f32 [I, J1, K, L] a, f32 [I, J2, K, L] b) {` |
| 15 | `benchmark/choreo/concat/3_attention_32xNx512x64_32xNx512x64_32xNx512x128.co` | `f95e37084e5e` | `__co__ f32 [I, J, K, L*2] ele_concat_K(f32 [I, J, K, L] a, f32 [I, J, K, L] b) {` |
| 16 | `benchmark/choreo/concat/4_dynamic_Nx256x56x56_Nx256x56x56_Nx512x56x56.co` | `588c1cf96f47` | `__co__ f32 [I, J_OUT, K, L] ele_concat_J(f32 [I, J1, K, L] a, f32 [I, J2, K, L] b) {` |
| 17 | `benchmark/choreo/concat/5_dynamic_Nx1280xHxW_Nx320xHxW_Nx1600xHxW.co` | `d2698fbd6069` | `__co__ f32 [I, J_OUT, K, L] ele_concat_J(f32 [I, J1, K, L] a, f32 [I, J2, K, L] b) {` |
| 18 | `benchmark/choreo/concat/6_dynamic_128xC1x112x112_128xC2x112x112_128xC1pC2x112x112.co` | `d8cbda8e456d` | `__co__ f32 [I, J_OUT, K, L] ele_concat_J(f32 [I, J1, K, L] a, f32 [I, J2, K, L] b) {` |
| 19 | `benchmark/choreo/concat/7_dynamic_32x197xE1_32x197xE2_32x197xE1pE2.co` | `4422b43ed575` | `__co__ f32 [I, J, L_OUT] ele_concat_J(f32 [I, J, L1] a, f32 [I, J, L2] b) {` |
| 20 | `benchmark/choreo/concat/8_dynamic_16x1024xD1_16x1024xD2_16x1024xD1pD2.co` | `26dadf42aef3` | `__co__ f32 [I, J, L_OUT] ele_concat_J(f32 [I, J, L1] a, f32 [I, J, L2] b) {` |
| 21 | `benchmark/choreo/concat/9_dynamic_64x128xHxW_64x128xHxW_64x256xHxW.co` | `8512ac750341` | `__co__ f32 [I, J_OUT, K, L] ele_concat_J(f32 [I, J1, K, L] a, f32 [I, J2, K, L] b) {` |
