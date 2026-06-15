module @f_19_transformer_32x512x2048_32x512x2048 {
  func.func @f_19_transformer_32x512x2048_32x512x2048(%input: tensor<32x512x2048xf32>) -> tensor<32x512x2048xf32> {
    %zero   = stablehlo.constant dense<0.0> : tensor<f32>
    %inf    = stablehlo.constant dense<3.4028235e+38> : tensor<f32>
    %result = stablehlo.clamp %zero, %input, %inf : (tensor<f32>, tensor<32x512x2048xf32>, tensor<f32>) -> tensor<32x512x2048xf32>
    return %result : tensor<32x512x2048xf32>
  }
}
