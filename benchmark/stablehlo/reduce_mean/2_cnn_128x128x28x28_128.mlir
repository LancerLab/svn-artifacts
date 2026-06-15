module @f_2_cnn_128x128x28x28_128 {
  func.func @f_2_cnn_128x128x28x28_128(%input: tensor<128x128x28x28xf32>) -> tensor<128xf32> {
    %zero     = stablehlo.constant dense<0.0> : tensor<f32>
    %sum_red  = stablehlo.reduce(%input init: %zero) across dimensions = [1, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
      reducer(%lhs: tensor<f32>, %rhs: tensor<f32>)  {
        %s = stablehlo.add %lhs, %rhs : tensor<f32>
        stablehlo.return %s : tensor<f32>
      }
    %nsz      = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %result   = stablehlo.divide %sum_red, %nsz : tensor<128xf32>
    return %result : tensor<128xf32>
  }
}
