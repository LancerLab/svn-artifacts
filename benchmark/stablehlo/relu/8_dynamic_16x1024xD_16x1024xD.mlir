module @f_8_dynamic_16x1024xD_16x1024xD {
  func.func @f_8_dynamic_16x1024xD_16x1024xD(%input: tensor<16x1024x?xf32>) -> tensor<16x1024x?xf32> {
    %zero   = stablehlo.constant dense<0.0> : tensor<f32>
    %inf    = stablehlo.constant dense<3.4028235e+38> : tensor<f32>
    %result = stablehlo.clamp %zero, %input, %inf : (tensor<f32>, tensor<16x1024x?xf32>, tensor<f32>) -> tensor<16x1024x?xf32>
    return %result : tensor<16x1024x?xf32>
  }
}
