# relu — operator settings

- **operator**: `relu`
- **reference semantics**: y = max(x, 0); elementwise, output shape = input shape.
- **cases**: 22 (derived from `benchmark/choreo/relu/`; signatures only, not source)

## Cases (provenance)

| # | source file | content-sha1 | operator signature |
|---|---|---|---|
| 1 | `benchmark/choreo/relu/10_dynamic_16x512xHxW_16x512xHxW.co` | `fece1067451b` | `__co__ auto ele_relu(f32 [I, J, HEIGHT, WIDTH] inp) {` |
| 2 | `benchmark/choreo/relu/11_dynamic_32xSx768_32xSx768.co` | `3fcc78801f03` | `__co__ auto ele_relu(f32 [I, NUM_HEADS, K, L] inp) {` |
| 3 | `benchmark/choreo/relu/12_dynamic_64xTx256_64xTx256.co` | `784bce983fe3` | `__co__ auto ele_relu(f32 [I, NUM_HEADS, K, L] inp) {` |
| 4 | `benchmark/choreo/relu/13_dynamic_32x512xV_32x512xV.co` | `c9aeaa17e170` | `__co__ auto ele_relu(f32 [I, J, EMBED_DIM, L] inp) {` |
| 5 | `benchmark/choreo/relu/14_efficientnet_64x1280x7x7_64x1280x7x7.co` | `cda8696e91ec` | `__co__ auto ele_relu(f32 [I, J, K, L] inp) {` |
| 6 | `benchmark/choreo/relu/15_gpt_16x1024x4096_16x1024x4096.co` | `c8b487c1647c` | `__co__ auto ele_relu(f32 [I, J, K, L] inp) {` |
| 7 | `benchmark/choreo/relu/16_lstm_64x100x256_64x100x256.co` | `977308a1cadc` | `__co__ auto ele_relu(f32 [I, J, K, L] inp) {` |
| 8 | `benchmark/choreo/relu/17_mobilenet_128x96x112x112_128x96x112x112.co` | `f61f1ef9bded` | `__co__ auto ele_relu(f32 [I, J, K, L] inp) {` |
| 9 | `benchmark/choreo/relu/18_resnet_64x256x56x56_64x256x56x56.co` | `8112643f2e44` | `__co__ auto ele_relu(f32 [I, J, K, L] inp) {` |
| 10 | `benchmark/choreo/relu/19_transformer_32x512x2048_32x512x2048.co` | `5c43625602d0` | `__co__ auto ele_relu(f32 [I, J, K, L] inp) {` |
| 11 | `benchmark/choreo/relu/1_bert_32x512x768_32x512x768.co` | `0493eb7f3672` | `__co__ auto ele_relu(f32 [I, J, K, L] inp) {` |
| 12 | `benchmark/choreo/relu/20_unet_16x512x32x32_16x512x32x32.co` | `add9c66dbad6` | `__co__ auto ele_relu(f32 [I, J, K, L] inp) {` |
| 13 | `benchmark/choreo/relu/21_vit_32x197x3072_32x197x3072.co` | `a2d2d907dcea` | `__co__ auto ele_relu(f32 [I, J, K, L] inp) {` |
| 14 | `benchmark/choreo/relu/2_cnn_128x128x28x28_128x128x28x28.co` | `b172db6f8237` | `__co__ auto ele_relu(f32 [I, J, K, L] inp) {` |
| 15 | `benchmark/choreo/relu/3_attention_32xNx512x64_32xNx512x64.co` | `71f4f0fece37` | `__co__ auto ele_relu(f32 [I, NUM_HEADS, K, L] inp) {` |
| 16 | `benchmark/choreo/relu/4_dynamic_Nx256x56x56_Nx256x56x56.co` | `6d1caf6a8dcb` | `__co__ auto ele_relu(f32 [NUM_HEADS, J, K, L] inp) {` |
| 17 | `benchmark/choreo/relu/5_dynamic_Nx1280xHxW_Nx1280xHxW.co` | `8562c18b72d5` | `__co__ auto ele_relu(f32 [NUM_HEADS, J, HEIGHT, WIDTH] inp) {` |
| 18 | `benchmark/choreo/relu/6_dynamic_128xCx112x112_128xCx112x112.co` | `bcf895321239` | `__co__ auto ele_relu(f32 [I, NUM_HEADS, K, L] inp) {` |
| 19 | `benchmark/choreo/relu/7_dynamic_32x197xE_32x197xE.co` | `08c3c6a8c8f4` | `__co__ auto ele_relu(f32 [I, J, EMBED_DIM, L] inp) {` |
| 20 | `benchmark/choreo/relu/8_dynamic_16x1024xD_16x1024xD.co` | `78ee29cc7d1d` | `__co__ auto ele_relu(f32 [I, NUM_HEADS, K, L] inp) {` |
| 21 | `benchmark/choreo/relu/9_dynamic_64x128xHxW_64x128xHxW.co` | `f3e085df9e50` | `__co__ auto ele_relu(f32 [I, J, HEIGHT, WIDTH] inp) {` |
| 22 | `benchmark/choreo/relu/bench_relu.co` | `fddb12793e7e` | `__co__ f32 [M, N] relu2d_view(f32 [M, N] input) {` |
