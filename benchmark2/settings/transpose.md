# transpose — operator settings

- **operator**: `transpose`
- **reference semantics**: y = permute(x, axes); output shape = permuted axes of input.
- **cases**: 21 (derived from `benchmark/choreo/transpose/`; signatures only, not source)

## Cases (provenance)

| # | source file | content-sha1 | operator signature |
|---|---|---|---|
| 1 | `benchmark/choreo/transpose/10_dynamic_16x512xHxW_16xHxWx512.co` | `1a77d49de078` | `__co__ auto TRANSPOSE(f32 [16, 512, scale_h, scale_w] i) {` |
| 2 | `benchmark/choreo/transpose/11_dynamic_32xSx768_32x768xS.co` | `ad5caaccd53f` | `__co__ auto TRANSPOSE(f32 [32, seq_len, 768] i) {` |
| 3 | `benchmark/choreo/transpose/12_dynamic_64xSx256_Sx64x256.co` | `ac21d8e674f9` | `__co__ auto TRANSPOSE(f32 [64, time_steps, 256] i) {` |
| 4 | `benchmark/choreo/transpose/13_dynamic_32x512xV_32xVx512.co` | `9046c51d32e9` | `__co__ auto TRANSPOSE(f32 [32, 512, vocab_size] i) {` |
| 5 | `benchmark/choreo/transpose/14_efficientnet_64x1280x7x7_64x7x7x1280.co` | `a76ffd951d10` | `__co__ auto TRANSPOSE(f32 [64, 1280, 7, 7] i) {` |
| 6 | `benchmark/choreo/transpose/15_gpt_16x16x1024x64_16x1024x16x64.co` | `7425696b3123` | `__co__ auto TRANSPOSE(f32 [16, 16, 1024, 64] i) {` |
| 7 | `benchmark/choreo/transpose/16_lstm_64x100x256_100x64x256.co` | `310ccac2537d` | `__co__ auto TRANSPOSE(f32 [64, 100, 256] i) {` |
| 8 | `benchmark/choreo/transpose/17_mobilenet_128xx96x112x112_128x112x112x96.co` | `b230ecab8226` | `__co__ auto TRANSPOSE(f32 [128, 96, 112, 112] i) {` |
| 9 | `benchmark/choreo/transpose/18_resnet_64x256x56x56_64x56x56x256.co` | `46b01705c423` | `__co__ auto TRANSPOSE(f32 [64, 256, 56, 56] i) {` |
| 10 | `benchmark/choreo/transpose/19_transformer_32x512x2048_32x2048x512.co` | `fda96507d856` | `__co__ auto TRANSPOSE(f32 [32, 512, 2048] i) {` |
| 11 | `benchmark/choreo/transpose/1_bert_32x512x768_32x768x512.co` | `43685ebe131d` | `__co__ auto TRANSPOSE(f32 [32, 512, 768] i) {` |
| 12 | `benchmark/choreo/transpose/20_unet_16x512x32x32_16x32x32x512.co` | `5ede5b49e57e` | `__co__ auto TRANSPOSE(f32 [16, 512, 32, 32] i) {` |
| 13 | `benchmark/choreo/transpose/21_vit_32x3x224x224_32x224x224x3.co` | `cec59088862b` | `__co__ auto TRANSPOSE(f32 [32, 3, 224, 224] i) {` |
| 14 | `benchmark/choreo/transpose/2_cnn_128x128x28x28_128x28x28x128.co` | `13fe42717e6f` | `__co__ auto TRANSPOSE(f32 [128, 128, 28, 28] i) {` |
| 15 | `benchmark/choreo/transpose/3_attention_32xNx512x64_32x512xNx64.co` | `be4b603506a6` | `__co__ auto TRANSPOSE(f32 [32, num_heads, 512, 64] i) {` |
| 16 | `benchmark/choreo/transpose/4_dynamic_Nx256x56x56_Nx56x56x256.co` | `ab0868c66876` | `__co__ auto TRANSPOSE(f32 [batch_size, 256, 56, 56] i) {` |
| 17 | `benchmark/choreo/transpose/5_dynamic_Nx1280xHxW_NxHxWx1280.co` | `f431457a46c1` | `__co__ auto TRANSPOSE(f32 [batch_size, 1280, height, width] i) {` |
| 18 | `benchmark/choreo/transpose/6_dynamic_128xCx112x112_128x112x112xC.co` | `974a0727419c` | `__co__ auto TRANSPOSE(f32 [128, channels, 112, 112] i) {` |
| 19 | `benchmark/choreo/transpose/7_dynamic_32x197xD_32xDx197.co` | `b58652686ce1` | `__co__ auto TRANSPOSE(f32 [32, 197, embed_dim] i) {` |
| 20 | `benchmark/choreo/transpose/8_dynamic_16x1024xD_16xDx1024.co` | `118286cf61a0` | `__co__ auto TRANSPOSE(f32 [16, 1024, d_model] i) {` |
| 21 | `benchmark/choreo/transpose/9_dynamic_64x128xHxW_64xHxWx128.co` | `f68eb4a712af` | `__co__ auto TRANSPOSE(f32 [64, 128, height, width] i) {` |
