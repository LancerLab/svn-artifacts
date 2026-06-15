module @f_13_dynamic_32x512xV_32x512xV {
  func.func @f_13_dynamic_32x512xV_32x512xV(%input: tensor<32x512x?xf32>) -> tensor<32x512x?xf32> {
    %zero   = stablehlo.constant dense<0.0> : tensor<f32>
    %inf    = stablehlo.constant dense<3.4028235e+38> : tensor<f32>
    %result = stablehlo.clamp %zero, %input, %inf : (tensor<f32>, tensor<32x512x?xf32>, tensor<f32>) -> tensor<32x512x?xf32>
    return %result : tensor<32x512x?xf32>
  }
}
