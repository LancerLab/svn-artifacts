module @f_6_dynamic_128xCx112x112_128xCx112x112 {
  func.func @f_6_dynamic_128xCx112x112_128xCx112x112(%input: tensor<128x?x112x112xf32>) -> tensor<128x?x112x112xf32> {
    %zero   = stablehlo.constant dense<0.0> : tensor<f32>
    %inf    = stablehlo.constant dense<3.4028235e+38> : tensor<f32>
    %result = stablehlo.clamp %zero, %input, %inf : (tensor<f32>, tensor<128x?x112x112xf32>, tensor<f32>) -> tensor<128x?x112x112xf32>
    return %result : tensor<128x?x112x112xf32>
  }
}
