module {
  func.func @f_20_unet_16x512x32x32_16x32x32x512(%input: tensor<16x512x32x32xf32>) -> tensor<16x32x32x512xf32> {
    %mid = tensor.collapse_shape %input [[0, 1, 2, 3]] : tensor<16x512x32x32xf32> into tensor<8388608xf32>
    %out = tensor.expand_shape %mid [[0, 1, 2, 3]] output_shape [16, 32, 32, 512] : tensor<8388608xf32> into tensor<16x32x32x512xf32>
    return %out : tensor<16x32x32x512xf32>
  }
}
