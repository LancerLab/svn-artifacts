# reshape — operator settings

- **operator**: `reshape`
- **reference semantics**: y = reshape(x, target); pure layout reinterpretation (element count preserved).
- **cases**: 21 (derived from `benchmark/choreo/reshape/`; signatures only, not source)

## Cases (provenance)

| # | source file | content-sha1 | operator signature |
|---|---|---|---|
| 1 | `benchmark/choreo/reshape/10_dynamic_64x128xHxW_64x128HW.co` | `13561a019521` | `__co__ auto reshape10(f32 [64, 128, height, width] input) {` |
| 2 | `benchmark/choreo/reshape/11_dynamic_16x512xSxS_16x512SS.co` | `4ef0c6169424` | `__co__ auto reshape11(f32 [16, 512, scale_h, scale_w] input) {` |
| 3 | `benchmark/choreo/reshape/12_dynamic_64xTx256_64Tx256.co` | `0f4000e4a5ec` | `__co__ auto reshape12(f32 [64, time_steps, 256] input) {` |
| 4 | `benchmark/choreo/reshape/13_dynamic_32x512xV_32x512V.co` | `5f36d9677aa1` | `__co__ auto reshape13(f32 [32, 512, vocab_size] input) {` |
| 5 | `benchmark/choreo/reshape/14_efficientnet_64x1280x7x7_64x62720.co` | `4f3494350bcd` | `__co__ auto reshape14(f32 [64, 1280, 7, 7] input) {` |
| 6 | `benchmark/choreo/reshape/15_gpt_16x1024x1024_16x16x1024x64.co` | `e90bf316d493` | `__co__ auto reshape15(f32 [16, 1024, 1024] input) {` |
| 7 | `benchmark/choreo/reshape/16_lstm_64x100x256_6400x256.co` | `be1588c6db03` | `__co__ auto reshape16(f32 [64, 100, 256] input) {` |
| 8 | `benchmark/choreo/reshape/17_mobilenet_128x96x7x7_128x4704.co` | `154bc75b753e` | `__co__ auto reshape17(f32 [128, 96, 7, 7] input) {` |
| 9 | `benchmark/choreo/reshape/18_resnet_64x512x7x7_64x25088.co` | `adabefc8328c` | `__co__ auto reshape18(f32 [64, 512, 7, 7] input) {` |
| 10 | `benchmark/choreo/reshape/19_transformer_32x512x2048_32x1048576.co` | `287b90e535fe` | `__co__ auto reshape19(f32 [32, 512, 2048] input) {` |
| 11 | `benchmark/choreo/reshape/1_bert_32x512x768_32x512x12x64.co` | `9078905004ef` | `__co__ auto reshape1(f32 [32, 512, 768] input) {` |
| 12 | `benchmark/choreo/reshape/20_unet_16x512x32x32_16x32x32x512.co` | `3d1fe46b6853` | `__co__ auto reshape20(f32 [16, 512, 32, 32] input) {` |
| 13 | `benchmark/choreo/reshape/21_vit_32x3x224x224_32x196x768.co` | `a8b41ffa1209` | `__co__ auto reshape21(f32 [32, 3, 224, 224] input) {` |
| 14 | `benchmark/choreo/reshape/2_cnn_128x128x28x28_128x100352.co` | `d6cff22e6fce` | `__co__ auto reshape2(f32 [128, 128, 28, 28] input) {` |
| 15 | `benchmark/choreo/reshape/3_attention_BxSx768_BxSx12x64.co` | `d3a5c97a4de4` | `__co__ auto reshape3(f32 [batch_size, seq_len, 768] input) {` |
| 16 | `benchmark/choreo/reshape/4_attention_32xNx512x64_32x512xN64.co` | `8ec9cbffede3` | `__co__ auto reshape4(f32 [32, num_heads, 512, 64] input) {` |
| 17 | `benchmark/choreo/reshape/5_dynamic_Bx1280xHxW_Bx1280HW.co` | `3506b2d4c194` | `__co__ auto reshape5(f32 [batch_size, 1280, height, width] input) {` |
| 18 | `benchmark/choreo/reshape/6_dynamic_128xCx112x112_128x112x112xC.co` | `bfafb576f5b0` | `__co__ auto reshape6(f32 [128, channels, 112, 112] input) {` |
| 19 | `benchmark/choreo/reshape/7_dynamic_32x197xE_32x197xEd64x64.co` | `b7f590d42188` | `__co__ auto reshape7(f32 [32, 197, embed_dim] input) {` |
| 20 | `benchmark/choreo/reshape/8_dynamic_16x1024xD_16x1024xDd64x64.co` | `247f12a20421` | `__co__ auto reshape8(f32 [16, 1024, d_model] input) {` |
| 21 | `benchmark/choreo/reshape/9_dynamic_Bx256xHxW_Bx256HW.co` | `43b1d9f5440f` | `__co__ auto reshape9(f32 [batch_size, 256, height, width] input) {` |
