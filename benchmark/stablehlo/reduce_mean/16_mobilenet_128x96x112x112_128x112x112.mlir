module @f_16_mobilenet_128x96x112x112_128x112x112 {
  func.func @f_16_mobilenet_128x96x112x112_128x112x112(%input: tensor<128x96x112x112xf32>) -> tensor<128x112x112xf32> {
    %zero     = stablehlo.constant dense<0.0> : tensor<f32>
    %sum_red  = stablehlo.reduce(%input init: %zero) across dimensions = [1] : (tensor<128x96x112x112xf32>, tensor<f32>) -> tensor<128x112x112xf32>
      reducer(%lhs: tensor<f32>, %rhs: tensor<f32>)  {
        %s = stablehlo.add %lhs, %rhs : tensor<f32>
        stablehlo.return %s : tensor<f32>
      }
    %nsz      = stablehlo.constant dense<96.0> : tensor<128x112x112xf32>
    %result   = stablehlo.divide %sum_red, %nsz : tensor<128x112x112xf32>
    return %result : tensor<128x112x112xf32>
  }
}
