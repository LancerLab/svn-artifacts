module @f_11_dynamic_32xSx768_32xSx768 {
  func.func @f_11_dynamic_32xSx768_32xSx768(%input: tensor<32x?x768xf32>) -> tensor<32x?x768xf32> {
    %zero   = stablehlo.constant dense<0.0> : tensor<f32>
    %inf    = stablehlo.constant dense<3.4028235e+38> : tensor<f32>
    %result = stablehlo.clamp %zero, %input, %inf : (tensor<f32>, tensor<32x?x768xf32>, tensor<f32>) -> tensor<32x?x768xf32>
    return %result : tensor<32x?x768xf32>
  }
}
