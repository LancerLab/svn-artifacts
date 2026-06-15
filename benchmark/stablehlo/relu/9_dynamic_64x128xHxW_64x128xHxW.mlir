module @f_9_dynamic_64x128xHxW_64x128xHxW {
  func.func @f_9_dynamic_64x128xHxW_64x128xHxW(%input: tensor<64x128x?x?xf32>) -> tensor<64x128x?x?xf32> {
    %zero   = stablehlo.constant dense<0.0> : tensor<f32>
    %inf    = stablehlo.constant dense<3.4028235e+38> : tensor<f32>
    %result = stablehlo.clamp %zero, %input, %inf : (tensor<f32>, tensor<64x128x?x?xf32>, tensor<f32>) -> tensor<64x128x?x?xf32>
    return %result : tensor<64x128x?x?xf32>
  }
}
