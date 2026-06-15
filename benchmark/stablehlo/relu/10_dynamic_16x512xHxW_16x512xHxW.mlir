module @f_10_dynamic_16x512xHxW_16x512xHxW {
  func.func @f_10_dynamic_16x512xHxW_16x512xHxW(%input: tensor<16x512x?x?xf32>) -> tensor<16x512x?x?xf32> {
    %zero   = stablehlo.constant dense<0.0> : tensor<f32>
    %inf    = stablehlo.constant dense<3.4028235e+38> : tensor<f32>
    %result = stablehlo.clamp %zero, %input, %inf : (tensor<f32>, tensor<16x512x?x?xf32>, tensor<f32>) -> tensor<16x512x?x?xf32>
    return %result : tensor<16x512x?x?xf32>
  }
}
