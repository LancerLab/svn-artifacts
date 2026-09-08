# sigmoid — operator settings

- **operator**: `sigmoid`
- **reference semantics**: y = 1/(1+exp(-x)); elementwise, output shape = input shape.
- **cases**: 21 (derived from `benchmark/choreo/sigmoid/`; signatures only, not source)

## Cases (provenance)

| # | source file | content-sha1 | operator signature |
|---|---|---|---|
| 1 | `benchmark/choreo/sigmoid/10_dynamic_16x512xHxW.co` | `5ca9a8745707` | `__co__ auto sigmoid(f32 [I, J, N0, N1] lhs) {` |
| 2 | `benchmark/choreo/sigmoid/11_dynamic_32xSx768.co` | `092090dc4e5c` | `__co__ auto sigmoid(f32 [I, N, K] lhs) {` |
| 3 | `benchmark/choreo/sigmoid/12_dynamic_64xTx256.co` | `582eb9e79dc5` | `__co__ auto sigmoid(f32 [I, N, K] lhs) {` |
| 4 | `benchmark/choreo/sigmoid/13_dynamic_32x512xV.co` | `7de64b2e640e` | `__co__ auto sigmoid(f32 [I, J, K] lhs) {` |
| 5 | `benchmark/choreo/sigmoid/14_efficientnet_64x1280x7x7.co` | `b897db538e7c` | `__co__ auto sigmoid(f32 [I, J, K, L] lhs) {` |
| 6 | `benchmark/choreo/sigmoid/15_gpt_16x1024x4096.co` | `7ecf4d3e9fa3` | `__co__ auto sigmoid(f32 [I, J, K] lhs) {` |
| 7 | `benchmark/choreo/sigmoid/16_lstm_64x100x256.co` | `ab636453af43` | `__co__ auto sigmoid(f32 [I, J, K] lhs) {` |
| 8 | `benchmark/choreo/sigmoid/17_mobilenet_128x96x112x112.co` | `bda411ca3e33` | `__co__ auto sigmoid(f32 [I, J, K, L] lhs) {` |
| 9 | `benchmark/choreo/sigmoid/18_resnet_64x256x56x56.co` | `350c8dba8282` | `__co__ auto sigmoid(f32 [I, J, K, L] lhs) {` |
| 10 | `benchmark/choreo/sigmoid/19_transformer_32x512x2048.co` | `3c379693cfc4` | `__co__ auto sigmoid(f32 [I, J, K] lhs) {` |
| 11 | `benchmark/choreo/sigmoid/1_bert_32x512x768.co` | `be96c3eabef0` | `__co__ auto sigmoid(f32 [I, J, K] lhs) {` |
| 12 | `benchmark/choreo/sigmoid/20_unet_16x512x32x32.co` | `bf326e3c39c9` | `__co__ auto sigmoid(f32 [I, J, K, L] lhs) {` |
| 13 | `benchmark/choreo/sigmoid/21_vit_32x197x3072.co` | `9209790cb56b` | `__co__ auto sigmoid(f32 [I, J, K] lhs) {` |
| 14 | `benchmark/choreo/sigmoid/2_cnn_128x128x28x28.co` | `79d91bbdd804` | `__co__ auto sigmoid(f32 [I, J, K, L] lhs) {` |
| 15 | `benchmark/choreo/sigmoid/3_attention_32xNx512x64.co` | `344092e5b3cf` | `__co__ auto sigmoid(f32 [I, N, K, L] lhs) {` |
| 16 | `benchmark/choreo/sigmoid/4_dynamic_Nx256x56x56.co` | `14ebe996f33e` | `__co__ auto sigmoid(f32 [N, J, K, L] lhs) {` |
| 17 | `benchmark/choreo/sigmoid/5_dynamic_Nx1280xMxK.co` | `af900714e6e5` | `__co__ auto sigmoid(f32 [N0, J, N1, N2] lhs) {` |
| 18 | `benchmark/choreo/sigmoid/6_dynamic_128xCx112x112.co` | `8ab94762b163` | `__co__ auto sigmoid(f32 [I, C, K, L] lhs) {` |
| 19 | `benchmark/choreo/sigmoid/7_dynamic_32x197xD.co` | `8be1274567ca` | `__co__ auto sigmoid(f32 [I, J, N] lhs) {` |
| 20 | `benchmark/choreo/sigmoid/8_dynamic_16x1024xD.co` | `6c6972a9dcff` | `__co__ auto sigmoid(f32 [I, J, K] lhs) {` |
| 21 | `benchmark/choreo/sigmoid/9_dynamic_64x128xHxW.co` | `cd1479164fa4` | `__co__ auto sigmoid(f32 [I, J, K, L] lhs) {` |
