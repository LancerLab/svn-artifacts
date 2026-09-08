# softmax — operator settings

- **operator**: `softmax`
- **reference semantics**: y = exp(x)/sum(exp(x), axis); normalized over the trailing axis; output = input shape.
- **cases**: 20 (derived from `benchmark/choreo/softmax/`; signatures only, not source)

## Cases (provenance)

| # | source file | content-sha1 | operator signature |
|---|---|---|---|
| 1 | `benchmark/choreo/softmax/10_dynamic_16x512xHxW_16x512xHxW.co` | `f4f5e7ad52da` | `__co__ auto softmax10(f32 [16, 512, scale_h, scale_w] input) {` |
| 2 | `benchmark/choreo/softmax/11_dynamic_32xSx768_32xSx768.co` | `8e4c95996cd4` | `__co__ auto softmax11(f32 [32, seq_len, 768] input) {` |
| 3 | `benchmark/choreo/softmax/12_dynamic_64xTx256_64xTx256.co` | `a00cc0528d05` | `__co__ auto softmax12(f32 [64, time_steps, 256] input) {` |
| 4 | `benchmark/choreo/softmax/13_dynamic_32x512xV_32x512xV.co` | `7939fb910d04` | `__co__ auto softmax13(f32 [32, 512, vocab_size] input) {` |
| 5 | `benchmark/choreo/softmax/14_efficientnet_64x1280x7x7_64x1280x7x7.co` | `df6d1e3c7cc6` | `__co__ auto softmax14(f32 [64, 1280, 7, 7] input) {` |
| 6 | `benchmark/choreo/softmax/15_gpt_16x1024x4096_16x1024x4096.co` | `837fb1cd0724` | `__co__ auto softmax15(f32 [16, 1024, 4096] input)  {` |
| 7 | `benchmark/choreo/softmax/16_lstm_64x100x256_64x100x256.co` | `e36c2a0cf752` | `__co__ auto softmax16(f32 [64, 100, 256] input) {` |
| 8 | `benchmark/choreo/softmax/17_mobilenet_128x96x112x112_128x96x112x112.co` | `77a16eddfb4d` | `__co__ auto softmax17(f32 [128, 96, 112, 112] input) {` |
| 9 | `benchmark/choreo/softmax/18_resnet_64x256x56x56_64x256x56x56.co` | `1371862de09f` | `__co__ auto softmax18(f32 [64, 256, 56, 56] input) {` |
| 10 | `benchmark/choreo/softmax/19_transformer_32x512x2048_32x512x2048.co` | `7110329a30e4` | `__co__ auto softmax19(f32 [32, 512, 2048] input)  {` |
| 11 | `benchmark/choreo/softmax/1_bert_32x512x768_32x512x768.co` | `2324c79a984c` | `__co__ auto softmax1(f32 [32, 512, 768] input) {` |
| 12 | `benchmark/choreo/softmax/20_vit_32x197x3072_32x197x3072.co` | `81c57bc03801` | `__co__ auto softmax20(f32 [32, 197, 3072] input) {` |
| 13 | `benchmark/choreo/softmax/2_cnn_128x128x28x28_128x128x28x28.co` | `8a43ad4f1259` | `__co__ auto softmax2(f32 [128, 128, 28, 28] input) {` |
| 14 | `benchmark/choreo/softmax/3_attention_32xNx512x64_32xNx512x64.co` | `3f8c9fee665e` | `__co__ auto softmax3(f32 [32, num_heads, 512, 64] input) {` |
| 15 | `benchmark/choreo/softmax/4_dynamic_Bx256x56x56_Bx256x56x56.co` | `b5f58ce47abd` | `__co__ auto softmax4(f32 [batch_size, 256, 56, 56] input)  {` |
| 16 | `benchmark/choreo/softmax/5_dynamic_Bx1280xHxW_Bx1280xHxW.co` | `508a4a1a375c` | `__co__ auto softmax5(f32 [batch_size, 1280, height, width] input) {` |
| 17 | `benchmark/choreo/softmax/6_dynamic_128xCx112x112_128xCx112x112.co` | `800594ef9f27` | `__co__ auto softmax6(f32 [128, channels, 112, 112] input) {` |
| 18 | `benchmark/choreo/softmax/7_dynamic_32x197xE_32x197xE.co` | `e03b9eac96d9` | `__co__ auto softmax7(f32 [32, 197, embed_dim] input) {` |
| 19 | `benchmark/choreo/softmax/8_dynamic_16x1024xD_16x1024xD.co` | `061ab74f3db7` | `__co__ auto softmax8(f32 [16, 1024, d_model] input) {` |
| 20 | `benchmark/choreo/softmax/9_dynamic_64x128xHxW_64x128xHxW.co` | `b3d3b7ed2a4c` | `__co__ auto softmax9(f32 [64, 128, height, width] input) {` |
