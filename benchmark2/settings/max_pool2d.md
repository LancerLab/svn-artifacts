# max_pool2d — operator settings

- **operator**: `max_pool2d`
- **reference semantics**: y = windowed max over spatial dims (kernel/stride per case); NCHW.
- **cases**: 20 (derived from `benchmark/choreo/max_pool2d/`; signatures only, not source)

## Cases (provenance)

| # | source file | content-sha1 | operator signature |
|---|---|---|---|
| 1 | `benchmark/choreo/max_pool2d/10_dynamic_32xCxHxW_32xCxHd5xWd5.co` | `9bea677e256b` | `__co__ auto max_pool2d10(f32 [32, channels, height, width] input) {` |
| 2 | `benchmark/choreo/max_pool2d/11_dynamic_16x512xSxS_16x512xSd2xSd2.co` | `c5a7010f9c35` | `__co__ auto max_pool2d11(f32 [16, 512, scale_h, scale_w] input) {` |
| 3 | `benchmark/choreo/max_pool2d/12_dynamic_Bx512xHxW_Bx512xHd2xWd2.co` | `de2e031d835c` | `__co__ auto max_pool2d12(f32 [batch_size, 512, height, width] input) {` |
| 4 | `benchmark/choreo/max_pool2d/13_dynamic_BxCx128x128_BxCx32x32.co` | `0c45a7894e7b` | `__co__ auto max_pool2d13(f32 [batch_size, channels, 128, 128] input) {` |
| 5 | `benchmark/choreo/max_pool2d/14_efficientnet_64x1280x14x14_64x1280x7x7.co` | `3532bee24440` | `__co__ auto max_pool2d14(f32 [64, 1280, 14, 14] input) {` |
| 6 | `benchmark/choreo/max_pool2d/15_inception_64x192x28x28_64x192x14x14.co` | `c262af0b969c` | `__co__ auto max_pool2d15(f32 [64, 192, 28, 28] input) {` |
| 7 | `benchmark/choreo/max_pool2d/16_mobilenet_128x96x224x224_128x96x112x112.co` | `4859c3025b58` | `__co__ auto max_pool2d16(f32 [128, 96, 224, 224] input) {` |
| 8 | `benchmark/choreo/max_pool2d/17_resnet_64x256x56x56_64x256x28x28.co` | `93d53e231b5c` | `__co__ auto max_pool2d17(f32 [64, 256, 56, 56] input) {` |
| 9 | `benchmark/choreo/max_pool2d/18_unet_16x512x64x64_16x512x32x32.co` | `9d50cf6dac64` | `__co__ auto max_pool2d18(f32 [16, 512, 64, 64] input) {` |
| 10 | `benchmark/choreo/max_pool2d/19_vgg_32x512x14x14_32x512x7x7.co` | `71f3d38ffe9d` | `__co__ auto max_pool2d19(f32 [32, 512, 14, 14] input) {` |
| 11 | `benchmark/choreo/max_pool2d/1_alexnet_64x256x27x27_64x256x13x13.co` | `a84968690266` | `__co__ auto max_pool2d1(f32 [64, 256, 27, 27] input) {` |
| 12 | `benchmark/choreo/max_pool2d/20_vit_32x3x224x224_32x3x112x112.co` | `0ff73f335fa9` | `__co__ auto max_pool2d20(f32 [32, 3, 224, 224] input) {` |
| 13 | `benchmark/choreo/max_pool2d/2_cnn_128x128x28x28_128x128x14x14.co` | `261c2f59aa60` | `__co__ auto max_pool2d2(f32 [128, 128, 28, 28] input) {` |
| 14 | `benchmark/choreo/max_pool2d/3_densenet_32x128x56x56_32x128x28x28.co` | `ab81c1225302` | `__co__ auto max_pool2d3(f32 [32, 128, 56, 56] input) {` |
| 15 | `benchmark/choreo/max_pool2d/4_dynamic_64xCxHxW_64xCxHd2xWd2.co` | `53c8b0c0c041` | `__co__ auto max_pool2d4(f32 [64, channels, height, width] input) {` |
| 16 | `benchmark/choreo/max_pool2d/5_dynamic_Bx256x56x56_Bx256x28x28.co` | `76a8e43f5641` | `__co__ auto max_pool2d5(f32 [batch_size, 256, 56, 56] input) {` |
| 17 | `benchmark/choreo/max_pool2d/6_dynamic_Bx1280xHxW_Bx1280xHd2xWd2.co` | `b04851d2dd18` | `__co__ auto max_pool2d6(f32 [batch_size, 1280, height, width] input) {` |
| 18 | `benchmark/choreo/max_pool2d/7_dynamic_128xCx112x112_128xCx56x56.co` | `2b3bf8937856` | `__co__ auto max_pool2d7(f32 [128, channels, 112, 112] input) {` |
| 19 | `benchmark/choreo/max_pool2d/8_dynamic_32xCxHxW_32xCxHd3xWd3.co` | `4fe3db879006` | `__co__ auto max_pool2d8(f32 [32, channels, height, width] input) {` |
| 20 | `benchmark/choreo/max_pool2d/9_dynamic_64x128xHxW_64x128xHd2xWd2.co` | `f48c01bb33a3` | `__co__ auto max_pool2d9(f32 [64, 128, height, width] input) {` |
