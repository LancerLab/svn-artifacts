module @f_20_unet_16x512x32x32_16x512x32x32_16x1024x32x32 {
  func.func @f_20_unet_16x512x32x32_16x512x32x32_16x1024x32x32(%in0: tensor<16x512x32x32xf32>, %in1: tensor<16x512x32x32xf32>) -> tensor<16x1024x32x32xf32> {
    %result = stablehlo.concatenate %in0, %in1, dim = 1 : (tensor<16x512x32x32xf32>, tensor<16x512x32x32xf32>) -> tensor<16x1024x32x32xf32>
    return %result : tensor<16x1024x32x32xf32>
  }
}
