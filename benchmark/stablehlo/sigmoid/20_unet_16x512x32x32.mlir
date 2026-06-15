module @f_20_unet_16x512x32x32 {
  func.func @f_20_unet_16x512x32x32(%input: tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32> {
    %result = stablehlo.logistic %input : tensor<16x512x32x32xf32>
    return %result : tensor<16x512x32x32xf32>
  }
}
