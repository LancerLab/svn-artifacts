module @f_14_efficientnet_64x1280x7x7_64x1280x7x7 {
  func.func @f_14_efficientnet_64x1280x7x7_64x1280x7x7(%input: tensor<64x1280x7x7xf32>) -> tensor<64x1280x7x7xf32> {
    %zero   = stablehlo.constant dense<0.0> : tensor<f32>
    %inf    = stablehlo.constant dense<3.4028235e+38> : tensor<f32>
    %result = stablehlo.clamp %zero, %input, %inf : (tensor<f32>, tensor<64x1280x7x7xf32>, tensor<f32>) -> tensor<64x1280x7x7xf32>
    return %result : tensor<64x1280x7x7xf32>
  }
}
