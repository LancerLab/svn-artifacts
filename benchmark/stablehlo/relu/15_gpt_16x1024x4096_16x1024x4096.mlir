module @f_15_gpt_16x1024x4096_16x1024x4096 {
  func.func @f_15_gpt_16x1024x4096_16x1024x4096(%input: tensor<16x1024x4096xf32>) -> tensor<16x1024x4096xf32> {
    %zero   = stablehlo.constant dense<0.0> : tensor<f32>
    %inf    = stablehlo.constant dense<3.4028235e+38> : tensor<f32>
    %result = stablehlo.clamp %zero, %input, %inf : (tensor<f32>, tensor<16x1024x4096xf32>, tensor<f32>) -> tensor<16x1024x4096xf32>
    return %result : tensor<16x1024x4096xf32>
  }
}
