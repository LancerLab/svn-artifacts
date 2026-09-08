# reduce_mean — operator settings

- **operator**: `reduce_mean`
- **reference semantics**: y = mean(x, axis); reduce one axis; output = x.shape with that axis dropped.
- **cases**: 20 (derived from `benchmark/choreo/reduce_mean/`; signatures only, not source)

## Cases (provenance)

| # | source file | content-sha1 | operator signature |
|---|---|---|---|
| 1 | `benchmark/choreo/reduce_mean/10_dynamic_32xSx768_32x768.co` | `4f15e42bf307` | `__co__ f32 [N, H] ele_reduce_mean(f32 [N, S, H] inp) {` |
| 2 | `benchmark/choreo/reduce_mean/11_dynamic_64xTx256_64x256.co` | `48a10ede9b44` | `__co__ f32 [N, D] ele_reduce_mean(f32 [N, N0, D] inp) {` |
| 3 | `benchmark/choreo/reduce_mean/12_dynamic_32x512xV_32x512.co` | `4998c368df87` | `__co__ f32 [N, S] ele_reduce_mean(f32 [N, S, V0] inp) {` |
| 4 | `benchmark/choreo/reduce_mean/13_efficientnet_64x1280x7x7_64x1280.co` | `d82ab576f462` | `__co__ f32 [N, C] ele_reduce_mean(f32 [N, C, H, W] inp) {` |
| 5 | `benchmark/choreo/reduce_mean/14_gpt_16x1024x1024_16x1024.co` | `0c61da18d383` | `__co__ f32 [N, H] ele_reduce_mean(f32 [N, S, H] inp) {` |
| 6 | `benchmark/choreo/reduce_mean/15_lstm_64x100x256_64x256.co` | `833b71f48904` | `__co__ f32 [N, D] ele_reduce_mean(f32 [N, T, D] inp) {` |
| 7 | `benchmark/choreo/reduce_mean/16_mobilenet_128x96x112x112_128x112x112.co` | `7afdefb8672d` | `__co__ f32 [N, H, W] ele_reduce_mean(f32 [N, C, H, W] inp) {` |
| 8 | `benchmark/choreo/reduce_mean/17_resnet_64x512x7x7_64x512.co` | `98a5d53ea781` | `__co__ f32 [N, C] ele_reduce_mean(f32 [N, C, H, W] inp) {` |
| 9 | `benchmark/choreo/reduce_mean/18_transformer_32x512x2048_32x2048.co` | `40fc684c1dbe` | `__co__ f32 [N, H] ele_reduce_mean(f32 [N, S, H] inp) {` |
| 10 | `benchmark/choreo/reduce_mean/19_unet_16x512x32x32_16x512x1x1.co` | `13bd0d1c64af` | `__co__ f32 [N, C, 1, 1] ele_reduce_mean(f32 [N, C, H, W] inp) {` |
| 11 | `benchmark/choreo/reduce_mean/1_bert_32x512x768_32x768.co` | `734ea98b78ce` | `__co__ f32 [I, H] ele_reduce_mean(f32 [I, S, H] inp) {` |
| 12 | `benchmark/choreo/reduce_mean/20_vit_32x197x768_32x768.co` | `a7cd4acb238b` | `__co__ f32 [N, H] ele_reduce_mean(f32 [N, P, H] inp) {` |
| 13 | `benchmark/choreo/reduce_mean/2_cnn_128x128x28x28_128.co` | `c83f05a7cbaa` | `__co__ f32 [B] ele_reduce_mean(f32 [B, C, H, W] inp) {` |
| 14 | `benchmark/choreo/reduce_mean/3_dynamic_Nx256x56x56_Nx256.co` | `fcff8ed177fe` | `__co__ auto ele_reduce_mean(f32 [N0, C, H, W] inp) {` |
| 15 | `benchmark/choreo/reduce_mean/4_dynamic_NxCxHxW_N.co` | `2d0104d08b1e` | `__co__ auto ele_reduce_mean(f32 [B0, C0, H0, W0] inp) {` |
| 16 | `benchmark/choreo/reduce_mean/5_dynamic_32xCx112x112_32x112x112.co` | `a00911855d28` | `__co__ f32 [N, H, W] ele_reduce_mean(f32 [N, C0, H, W] inp) {` |
| 17 | `benchmark/choreo/reduce_mean/6_dynamic_16x512xE_16x512.co` | `1f16a8d48766` | `__co__ f32 [N, S] ele_reduce_mean(f32 [N, S, E0] inp) {` |
| 18 | `benchmark/choreo/reduce_mean/7_dynamic_64x128xHxW_64x128.co` | `181ca03a3e1c` | `__co__ f32 [N, C] ele_reduce_mean(f32 [N, C, H, W] inp) {` |
| 19 | `benchmark/choreo/reduce_mean/8_dynamic_16x512xHxW_16x512.co` | `01fb65790208` | `__co__ f32 [N, S] ele_reduce_mean(f32 [N, S, H, W] inp) {` |
| 20 | `benchmark/choreo/reduce_mean/9_dynamic_32xPx768_32x768.co` | `11fdd4706f41` | `__co__ f32 [N, H] ele_reduce_mean(f32 [N, P, H] inp) {` |
