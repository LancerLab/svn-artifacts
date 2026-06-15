module @f_7_dynamic_32x197xE_32x197xE {
  func.func @f_7_dynamic_32x197xE_32x197xE(%input: tensor<32x197x?xf32>) -> tensor<32x197x?xf32> {
    %zero   = stablehlo.constant dense<0.0> : tensor<f32>
    %inf    = stablehlo.constant dense<3.4028235e+38> : tensor<f32>
    %result = stablehlo.clamp %zero, %input, %inf : (tensor<f32>, tensor<32x197x?xf32>, tensor<f32>) -> tensor<32x197x?xf32>
    return %result : tensor<32x197x?xf32>
  }
}
