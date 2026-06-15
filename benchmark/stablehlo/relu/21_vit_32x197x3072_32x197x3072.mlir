module @f_21_vit_32x197x3072_32x197x3072 {
  func.func @f_21_vit_32x197x3072_32x197x3072(%input: tensor<32x197x3072xf32>) -> tensor<32x197x3072xf32> {
    %zero   = stablehlo.constant dense<0.0> : tensor<f32>
    %inf    = stablehlo.constant dense<3.4028235e+38> : tensor<f32>
    %result = stablehlo.clamp %zero, %input, %inf : (tensor<f32>, tensor<32x197x3072xf32>, tensor<f32>) -> tensor<32x197x3072xf32>
    return %result : tensor<32x197x3072xf32>
  }
}
