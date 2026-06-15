module @f_20_unet_16x512x32x32_16x32x32x512 {
  func.func @f_20_unet_16x512x32x32_16x32x32x512(%input: tensor<16x512x32x32xf32>) -> tensor<16x32x32x512xf32> {
    %result = stablehlo.transpose %input, dims = [0, 2, 3, 1] : (tensor<16x512x32x32xf32>) -> tensor<16x32x32x512xf32>
    return %result : tensor<16x32x32x512xf32>
  }
}
