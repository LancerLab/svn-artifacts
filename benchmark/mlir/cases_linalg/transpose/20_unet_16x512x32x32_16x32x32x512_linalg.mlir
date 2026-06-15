module {
  func.func @f_20_unet_16x512x32x32_16x32x32x512_linalg(%input: tensor<16x512x32x32xf32>) -> tensor<16x32x32x512xf32> {
    %init = tensor.empty() : tensor<16x32x32x512xf32>
    %r = linalg.transpose ins(%input : tensor<16x512x32x32xf32>) outs(%init : tensor<16x32x32x512xf32>) permutation = [0, 2, 3, 1]
    return %r : tensor<16x32x32x512xf32>
  }
}
