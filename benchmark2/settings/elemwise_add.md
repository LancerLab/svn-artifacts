# elemwise_add — operator settings

- **operator**: `elemwise_add`
- **reference semantics**: y = lhs + rhs; elementwise with broadcast (shapes equal or broadcastable).
- **cases**: 21 (derived from `benchmark/choreo/elemwise_add/`; signatures only, not source)

## Cases (provenance)

| # | source file | content-sha1 | operator signature |
|---|---|---|---|
| 1 | `benchmark/choreo/elemwise_add/10_dynamic_16x512xHxW_16x512xHxW_16x512xHxW.co` | `bd41f0cc2901` | `__co__ auto ele_add(f32 [I, J, N0, N1] lhs, f32 [I, J, N0, N1] rhs) {` |
| 2 | `benchmark/choreo/elemwise_add/11_dynamic_32xSx768_32xSx768_32xSx768.co` | `82122335e90b` | `__co__ auto ele_add(f32 [I, N, K] lhs, f32 [I, N, K] rhs) {` |
| 3 | `benchmark/choreo/elemwise_add/12_dynamic_64xTx256_64xTx256_64xTx256.co` | `17fa7757961e` | `__co__ auto ele_add(f32 [I, N, K] lhs, f32 [I, N, K] rhs) {` |
| 4 | `benchmark/choreo/elemwise_add/13_dynamic_32x512xV_32x512xV_32x512xV.co` | `1f9f43635a25` | `__co__ auto ele_add(f32 [I, J, N] lhs, f32 [I, J, N] rhs) {` |
| 5 | `benchmark/choreo/elemwise_add/14_efficientnet_64x1280x7x7_64x1280x7x7_64x1280x7x7.co` | `0a3b42bde174` | `__co__ auto ele_add(f32 [I, J, K, L] lhs, f32 [I, J, K, L] rhs) {` |
| 6 | `benchmark/choreo/elemwise_add/15_gpt_16x1024x4096_16x1024x4096_16x1024x4096.co` | `d3577455e947` | `__co__ auto ele_add(f32 [I, J, K] lhs, f32 [I, J, K] rhs) {` |
| 7 | `benchmark/choreo/elemwise_add/16_lstm_64x100x256_64x100x256_64x100x256.co` | `daf69ad5df82` | `__co__ auto ele_add(f32 [I, J, K] lhs, f32 [I, J, K] rhs) {` |
| 8 | `benchmark/choreo/elemwise_add/17_mobilenet_128x96x112x112_128x96x112x112_128x96x112x112.co` | `bb99de882dbd` | `__co__ auto ele_add(f32 [I, J, K, L] lhs, f32 [I, J, K, L] rhs) {` |
| 9 | `benchmark/choreo/elemwise_add/18_resnet_64x256x56x56_64x256x56x56_64x256x56x56.co` | `473965986f57` | `__co__ auto ele_add(f32 [I, J, K, L] lhs, f32 [I, J, K, L] rhs) {` |
| 10 | `benchmark/choreo/elemwise_add/19_transformer_32x512x2048_32x512x2048_32x512x2048.co` | `cf53562b999a` | `__co__ auto ele_add(f32 [I, J, K] lhs, f32 [I, J, K] rhs) {` |
| 11 | `benchmark/choreo/elemwise_add/1_bert_32x512x768_32x512x768_32x512x768.co` | `2f84da736307` | `__co__ auto ele_add(f32 [I, J, K] lhs, f32 [I, J, K] rhs) {` |
| 12 | `benchmark/choreo/elemwise_add/20_unet_16x512x32x32_16x512x32x32_16x512x32x32.co` | `d2626d98db34` | `__co__ auto ele_add(f32 [I, J, K, L] lhs, f32 [I, J, K, L] rhs) {` |
| 13 | `benchmark/choreo/elemwise_add/21_vit_32x197x768_32x197x768_32x197x768.co` | `148807964b64` | `__co__ auto ele_add(f32 [I, J, K] lhs, f32 [I, J, K] rhs) {` |
| 14 | `benchmark/choreo/elemwise_add/2_broadcast_128x256x28x28_256_128x256x28x28.co` | `9c8ed27d468b` | `__co__ auto ele_add(f32 [I, J, K, L] lhs, f32 [J] rhs) {` |
| 15 | `benchmark/choreo/elemwise_add/3_broadcast_32x512x768_1_32x512x768.co` | `9be6bafd88b8` | `__co__ auto ele_add(f32 [I, J, K] lhs, f32 rhs) {` |
| 16 | `benchmark/choreo/elemwise_add/4_attention_32xNx512x64_32xNx512x64_32xNx512x64.co` | `1226bd874a93` | `__co__ auto ele_add(f32 [I, N, K, L] lhs, f32 [I, N, K, L] rhs) {` |
| 17 | `benchmark/choreo/elemwise_add/5_dynamic_Nx256x56x56_Nx256x56x56_Nx256x56x56.co` | `fb5d5e81d114` | `__co__ auto ele_add(f32 [N, J, K, L] lhs, f32 [N, J, K, L] rhs) {` |
| 18 | `benchmark/choreo/elemwise_add/6_dynamic_Nx1280xMxK_Nx1280x1x1_Nx1280xMxK.co` | `cd3748ccf102` | `__co__ auto ele_add(f32 [N0, J, N1, N2] lhs, f32 [N0, J] rhs) {` |
| 19 | `benchmark/choreo/elemwise_add/7_dynamic_128xCx112x112_128xCx112x112_128xCx112x112.co` | `f062e128a61c` | `__co__ auto ele_add(f32 [I, N, K, L] lhs, f32 [I, N, K, L] rhs) {` |
| 20 | `benchmark/choreo/elemwise_add/8_dynamic_16x1024xD_16x1024xD_16x1024xD.co` | `0a9b044258c0` | `__co__ auto ele_add(f32 [I, J, N] lhs, f32 [I, J, N] rhs) {` |
| 21 | `benchmark/choreo/elemwise_add/9_dynamic_64x128xWxH_64x128xWxH_64x128xWxH.co` | `01966b3650d1` | `__co__ auto ele_add(f32 [I, J, N0, N1] lhs, f32 [I, J, N0, N1] rhs) {` |
