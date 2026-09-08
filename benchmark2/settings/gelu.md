# gelu — operator settings

- **operator**: `gelu`
- **reference semantics**: y = 0.5*x*(1+erf(x/sqrt(2))); elementwise, output shape = input shape.
- **cases**: 21 (derived from `benchmark/choreo/gelu/`; signatures only, not source)

## Cases (provenance)

| # | source file | content-sha1 | operator signature |
|---|---|---|---|
| 1 | `benchmark/choreo/gelu/10_dynamic_16x512xHxW.co` | `1933492d44d1` | `__co__ auto GELU(f32 [16, 512, scale_h, scale_w] i) {` |
| 2 | `benchmark/choreo/gelu/11_dynamic_32xSx768.co` | `db82b0696085` | `__co__ auto GELU(f32 [32, seq_len, 768] i) {` |
| 3 | `benchmark/choreo/gelu/12_dynamic_64xTx256.co` | `d15f851d1589` | `__co__ auto GELU(f32 [64, time_steps, 256] i) {` |
| 4 | `benchmark/choreo/gelu/13_dynamic_32x512xV.co` | `9b73c4e2fdce` | `__co__ auto GELU(f32 [32, 512, vocab_size] i) {` |
| 5 | `benchmark/choreo/gelu/14_efficientnet_64x1280x7x7.co` | `390e5a1c2ff4` | `__co__ auto GELU(f32 [64, 1280, 7, 7] i) {` |
| 6 | `benchmark/choreo/gelu/15_gpt_16x1024x4096.co` | `475e9b0976a0` | `__co__ auto GELU(f32 [16, 1024, 4096] i) {` |
| 7 | `benchmark/choreo/gelu/16_lstm_64x100x256.co` | `c42fd9a7a1d1` | `__co__ auto GELU(f32 [64, 100, 256] i) {` |
| 8 | `benchmark/choreo/gelu/17_mobilenet_128x96x112x112.co` | `d62c7e6ff078` | `__co__ auto GELU(f32 [128, 96, 112, 112] i) {` |
| 9 | `benchmark/choreo/gelu/18_resnet_64x256x56x56.co` | `c0524be57df5` | `__co__ auto GELU(f32 [64, 256, 56, 56] i) {` |
| 10 | `benchmark/choreo/gelu/19_transformer_32x512x2048.co` | `2c879e2f0b00` | `__co__ auto GELU(f32 [32, 512, 2048] i) {` |
| 11 | `benchmark/choreo/gelu/1_bert_32x512x768.co` | `c1693cfd51b9` | `__co__ auto GELU(f32 [32, 512, 768] i) {` |
| 12 | `benchmark/choreo/gelu/20_unet_16x512x32x32.co` | `14774b4cff4e` | `__co__ auto GELU(f32 [16, 512, 32, 32] i) {` |
| 13 | `benchmark/choreo/gelu/21_vit_32x197x3072.co` | `495b30102d9c` | `__co__ auto GELU(f32 [32, 197, 3072] i) {` |
| 14 | `benchmark/choreo/gelu/2_cnn_128x128x28x28.co` | `0ac27f21d6dc` | `__co__ auto GELU(f32 [128, 128, 28, 28] i) {` |
| 15 | `benchmark/choreo/gelu/3_attention_32xNx512x64.co` | `09d5bf0f45e4` | `__co__ auto GELU(f32 [32, num_heads, 512, 64] i) {` |
| 16 | `benchmark/choreo/gelu/4_dynamic_Nx256x56x56.co` | `4335313ed724` | `__co__ auto GELU(f32 [batch_size, 256, 56, 56] i) {` |
| 17 | `benchmark/choreo/gelu/5_dynamic_Nx1280xHxW.co` | `c8f1d79a8c57` | `__co__ auto GELU(f32 [batch_size, 1280, height, width] i) {` |
| 18 | `benchmark/choreo/gelu/6_dynamic_128xCx112x112.co` | `2c6a93755e37` | `__co__ auto GELU(f32 [128, channels, 112, 112] i) {` |
| 19 | `benchmark/choreo/gelu/7_dynamic_32x197xD.co` | `8a7cb5e31e8b` | `__co__ auto GELU(f32 [32, 197, embed_dim] i) {` |
| 20 | `benchmark/choreo/gelu/8_dynamic_16x1024xD.co` | `9f5c0c1c8f35` | `__co__ auto GELU(f32 [16, 1024, d_model] i) {` |
| 21 | `benchmark/choreo/gelu/9_dynamic_64x128xHxW.co` | `29b270cc918a` | `__co__ auto GELU(f32 [64, 128, height, width] i) {` |
