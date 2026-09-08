# layer_normalization — operator settings

- **operator**: `layer_normalization`
- **reference semantics**: y = (x - mean)/sqrt(var+eps) * scale + bias; mean/var reduced over the trailing (normalized) dims; scale/bias shaped to those dims.
- **cases**: 21 (derived from `benchmark/choreo/layer_normalization/`; signatures only, not source)

## Cases (provenance)

| # | source file | content-sha1 | operator signature |
|---|---|---|---|
| 1 | `benchmark/choreo/layer_normalization/10_dynamic_16x512xHxW_HxW_HxW.co` | `555c63559c99` | `__co__ auto layer_norm(f32 [I, J, N1, N2] lhs, f32 [N1, N2] scale, f32 [N1, N2] bias) {` |
| 2 | `benchmark/choreo/layer_normalization/11_dynamic_32xSx768_768_768.co` | `3c2a924a69bd` | `__co__ auto layer_norm(f32 [I, N0, K] lhs, f32 [K] scale, f32 [K] bias) {` |
| 3 | `benchmark/choreo/layer_normalization/12_dynamic_64xTx256_256_256.co` | `5d28ce50e633` | `__co__ auto layer_norm(f32 [I, N0, K] lhs, f32 [K] scale, f32 [K] bias) {` |
| 4 | `benchmark/choreo/layer_normalization/13_dynamic_32x512xV_V_V.co` | `46decec6bb57` | `__co__ auto layer_norm(f32 [I, J, N0] lhs, f32 [N0] scale, f32 [N0] bias) {` |
| 5 | `benchmark/choreo/layer_normalization/14_efficientnet_64x1280x7x7_7x7_7x7.co` | `b8baa2537e9b` | `__co__ auto layer_norm(f32 [I, J, K, L] lhs, f32 [K, L] scale, f32 [K, L] bias) {` |
| 6 | `benchmark/choreo/layer_normalization/15_gpt_16x1024x4096_4096_4096.co` | `4b9d8e13afb2` | `__co__ auto layer_norm(f32 [I, J, K] lhs, f32 [K] scale, f32 [K] bias) {` |
| 7 | `benchmark/choreo/layer_normalization/16_lstm_64x100x256_256_256.co` | `020e847cdcfb` | `__co__ auto layer_norm(f32 [I, J, K] lhs, f32 [K] scale, f32 [K] bias) {` |
| 8 | `benchmark/choreo/layer_normalization/17_mobilenet_128x96x112x112_112x112_112x112.co` | `738e4646a876` | `__co__ auto layer_norm(f32 [I, J, K, L] lhs, f32 [K, L] scale, f32 [K, L] bias) {` |
| 9 | `benchmark/choreo/layer_normalization/18_resnet_64x256x56x56_56x56_56x56.co` | `040a4e2e89fe` | `__co__ auto layer_norm(f32 [I, J, K, L] lhs, f32 [K, L] scale, f32 [K, L] bias) {` |
| 10 | `benchmark/choreo/layer_normalization/19_transformer_32x512x2048_2048_2048.co` | `fd4be4ab3d1b` | `__co__ auto layer_norm(f32 [I, J, K] lhs, f32 [K] scale, f32 [K] bias) {` |
| 11 | `benchmark/choreo/layer_normalization/1_bert_32x512x768_768_768.co` | `ae22fd957dba` | `__co__ auto layer_norm(f32 [I, J, K] lhs, f32 [K] scale, f32 [K] bias) {` |
| 12 | `benchmark/choreo/layer_normalization/20_unet_16x512x32x32_32x32_32x32.co` | `c134af486436` | `__co__ auto layer_norm(f32 [I, J, K, L] lhs, f32 [K, L] scale, f32 [K, L] bias) {` |
| 13 | `benchmark/choreo/layer_normalization/21_vit_32x197x3072_3072_3072.co` | `a4683d0b78a8` | `__co__ auto layer_norm(f32 [I, J, K] lhs, f32 [K] scale, f32 [K] bias) {` |
| 14 | `benchmark/choreo/layer_normalization/2_cnn_128x128x28x28_28x28_28x28.co` | `66bec9934b65` | `__co__ auto layer_norm(f32 [I, J, K, L] lhs, f32 [K, L] scale, f32 [K, L] bias) {` |
| 15 | `benchmark/choreo/layer_normalization/3_attention_32xNx512x64_64_64.co` | `62ee3052460c` | `__co__ auto layer_norm(f32 [I, N0, K, L] lhs, f32 [L] scale, f32 [L] bias) {` |
| 16 | `benchmark/choreo/layer_normalization/4_dynamic_Nx256x56x56_56x56_56x56.co` | `93e73ecb8024` | `__co__ auto layer_norm(f32 [N0, J, K, L] lhs, f32 [K, L] scale, f32 [K, L] bias) {` |
| 17 | `benchmark/choreo/layer_normalization/5_dynamic_Nx1280xMxK_MxK_MxK.co` | `700aa9210cd8` | `__co__ auto layer_norm(f32 [N0, J, N1, N2] lhs, f32 [N1, N2] scale, f32 [N1, N2] bias) {` |
| 18 | `benchmark/choreo/layer_normalization/6_dynamic_128xCx112x112_112x112_112x112.co` | `b797da4202a1` | `__co__ auto layer_norm(f32 [I, N0, K, L] lhs, f32 [K, L] scale, f32 [K, L] bias) {` |
| 19 | `benchmark/choreo/layer_normalization/7_dynamic_32x197xD_D_D.co` | `6e8595f25059` | `__co__ auto layer_norm(f32 [I, J, N0] lhs, f32 [N0] scale, f32 [N0] bias) {` |
| 20 | `benchmark/choreo/layer_normalization/8_dynamic_16x1024xD_D_D.co` | `1c6fe5b3f877` | `__co__ auto layer_norm(f32 [I, J, N0] lhs, f32 [N0] scale, f32 [N0] bias) {` |
| 21 | `benchmark/choreo/layer_normalization/9_dynamic_64x128xHxW_HxW_HxW.co` | `5e03270dbe03` | `__co__ auto layer_norm(f32 [I, J, N1, N2] lhs, f32 [N1, N2] scale, f32 [N1, N2] bias) {` |
