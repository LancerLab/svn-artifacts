module @f_20_unet_16x512x32x32_16x512x32x32 {
  func.func @f_20_unet_16x512x32x32_16x512x32x32(%input: tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32> {
    %zero   = stablehlo.constant dense<0.0> : tensor<f32>
    %inf    = stablehlo.constant dense<3.4028235e+38> : tensor<f32>
    %result = stablehlo.clamp %zero, %input, %inf : (tensor<f32>, tensor<16x512x32x32xf32>, tensor<f32>) -> tensor<16x512x32x32xf32>
    return %result : tensor<16x512x32x32xf32>
  }
}
