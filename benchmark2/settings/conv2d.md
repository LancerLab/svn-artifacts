# conv2d — operator settings

- **operator**: `conv2d`
- **reference semantics**: y = conv2d(x, w, stride, padding, dilation); standard NCHW 2-D convolution (optionally strided/padded/dilated).
- **cases**: 21 (derived from `benchmark/choreo/conv2d/`; signatures only, not source)

## Cases (provenance)

| # | source file | content-sha1 | operator signature |
|---|---|---|---|
| 1 | `benchmark/choreo/conv2d/10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D.co` | `d434e35e9aee` | `__co__ auto CONV2D(f32 [8, 128, 16, 16] i, f32 [256, 128, 3, 3] w, int stride, int padding, int dilation) {` |
| 2 | `benchmark/choreo/conv2d/11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1.co` | `e0475836ac2a` | `__co__ auto CONV2D(f32 [64, 128, 32, 32] i, f32 [32, 128, 1, 1] w) {` |
| 3 | `benchmark/choreo/conv2d/12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1.co` | `5f25886ae095` | `__co__ auto CONV2D(f32 [64, 40, 56, 56] i, f32 [240, 40, 1, 1] w) {` |
| 4 | `benchmark/choreo/conv2d/13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1.co` | `ec97c2095317` | `__co__ auto CONV2D(f32 [32, 192, 28, 28] i, f32 [64, 192, 1, 1] w) {` |
| 5 | `benchmark/choreo/conv2d/14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D.co` | `ab40bb948a1e` | `__co__ auto CONV2D(f32 [128, 32, 112, 112] i, f32 [32, 32, 3, 3] w, int stride, int padding, int dilation) {` |
| 6 | `benchmark/choreo/conv2d/15_resnet_32x256x56x56_64x256x1x1_32x64x56x56_1_0_1.co` | `33f6e5b87bcb` | `__co__ auto CONV2D(f32 [32, 256, 56, 56] i, f32 [64, 256, 1, 1] w) {` |
| 7 | `benchmark/choreo/conv2d/16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D.co` | `720a45b1a52c` | `__co__ auto CONV2D(f32 [64, 3, 224, 224] i, f32 [64, 3, 7, 7] w, int stride, int padding, int dilation) {` |
| 8 | `benchmark/choreo/conv2d/17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1.co` | `c5b5632fbe23` | `__co__ auto CONV2D(f32 [8, 256, 64, 64] i, f32 [21, 256, 1, 1] w) {` |
| 9 | `benchmark/choreo/conv2d/18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D.co` | `82d8f4b10641` | `__co__ auto CONV2D(f32 [16, 64, 128, 128] i, f32 [128, 64, 3, 3] w, int stride, int padding, int dilation) {` |
| 10 | `benchmark/choreo/conv2d/19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1.co` | `dc3fb5c60b45` | `__co__ auto CONV2D(f32 [32, 3, 224, 224] i, f32 [768, 3, 16, 16] w) {` |
| 11 | `benchmark/choreo/conv2d/1_attention_32x512xHxW_1024x512x1x1_32x1024xHxW_1_0_1.co` | `18cd8e2a6cdc` | `__co__ auto CONV2D(f32 [32, 512, attn_h, attn_w] i, f32 [1024, 512, 1, 1] w) {` |
| 12 | `benchmark/choreo/conv2d/20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1.co` | `878d1d657771` | `__co__ auto CONV2D(f32 [16, 1024, 13, 13] i, f32 [255, 1024, 1, 1] w) {` |
| 13 | `benchmark/choreo/conv2d/21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D.co` | `cd6b5c8e9ca1` | `__co__ auto CONV2D(f32 [32, 768, 14, 14] i, f32 [1000, 768, 7, 7] w, int stride, int padding, int dilation) {` |
| 14 | `benchmark/choreo/conv2d/2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D.co` | `b502bd0bc1fe` | `__co__ auto CONV2D(f32 [batch_size, 64, 56, 56] i, f32 [128, 64, 3, 3] w, int stride, int padding, int dilation) {` |
| 15 | `benchmark/choreo/conv2d/3_dynamic_16x256xHxW_256x256x3x3_16x256xHxW_S_P_D.co` | `959ff3eb429a` | `__co__ auto CONV2D(f32 [16, 256, pyramid_h, pyramid_w] i, f32 [256, 256, 3, 3] w, int stride, int padding, int dilation) {` |
| 16 | `benchmark/choreo/conv2d/4_dynamic_32x128xHxW_256x128x3x3_32x256xHxW_S_P_D.co` | `27da0335671c` | `__co__ auto CONV2D(f32 [32, 128, height, width] i, f32 [256, 128, 3, 3] w, int stride, int padding, int dilation) {` |
| 17 | `benchmark/choreo/conv2d/5_dynamic_128xCx112x112_64xCx1x1_128x64x112x112_1_0_1.co` | `1ce2d0d46804` | `__co__ auto CONV2D(f32 [128, C, 112, 112] i, f32 [64, C, 1, 1] w) {` |
| 18 | `benchmark/choreo/conv2d/6_dynamic_64x64x56x56_128x64x3x3_64x128x56x56_S_P_D.co` | `2b775a866af9` | `__co__ auto CONV2D(f32 [64, 64, 56, 56] i, f32 [128, 64, 3, 3] w, int stride, int padding, int dilation) {` |
| 19 | `benchmark/choreo/conv2d/7_dynamic_8x256xHxW_512x256x1x1_8x512xHxW_1_0_1.co` | `f7ba0b71202e` | `__co__ auto CONV2D(f32 [8, 256, scale_h, scale_w] i, f32 [512, 256, 1, 1] w) {` |
| 20 | `benchmark/choreo/conv2d/8_dynamic_64x256x28x28_Cx256x3x3_64xCx28x28_S_P_D.co` | `1a7565ea2aed` | `__co__ auto CONV2D(f32 [64, 256, 28, 28] i, f32 [128, 256, 3, 3] w, int stride, int padding, int dilation) {` |
| 21 | `benchmark/choreo/conv2d/9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D.co` | `bf99f09c0ffe` | `__co__ auto CONV2D(f32 [16, 64, 224, 224] i, f32 [128, 64, 7, 7] w, int stride, int padding, int dilation) {` |
