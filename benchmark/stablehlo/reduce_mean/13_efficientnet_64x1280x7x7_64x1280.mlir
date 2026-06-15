module @f_13_efficientnet_64x1280x7x7_64x1280 {
  func.func @f_13_efficientnet_64x1280x7x7_64x1280(%input: tensor<64x1280x7x7xf32>) -> tensor<64x1280xf32> {
    %zero     = stablehlo.constant dense<0.0> : tensor<f32>
    %sum_red  = stablehlo.reduce(%input init: %zero) across dimensions = [2, 3] : (tensor<64x1280x7x7xf32>, tensor<f32>) -> tensor<64x1280xf32>
      reducer(%lhs: tensor<f32>, %rhs: tensor<f32>)  {
        %s = stablehlo.add %lhs, %rhs : tensor<f32>
        stablehlo.return %s : tensor<f32>
      }
    %nsz      = stablehlo.constant dense<49.0> : tensor<64x1280xf32>
    %result   = stablehlo.divide %sum_red, %nsz : tensor<64x1280xf32>
    return %result : tensor<64x1280xf32>
  }
}
