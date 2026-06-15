module @f_15_lstm_64x100x256_64x256 {
  func.func @f_15_lstm_64x100x256_64x256(%input: tensor<64x100x256xf32>) -> tensor<64x256xf32> {
    %zero     = stablehlo.constant dense<0.0> : tensor<f32>
    %sum_red  = stablehlo.reduce(%input init: %zero) across dimensions = [1] : (tensor<64x100x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      reducer(%lhs: tensor<f32>, %rhs: tensor<f32>)  {
        %s = stablehlo.add %lhs, %rhs : tensor<f32>
        stablehlo.return %s : tensor<f32>
      }
    %nsz      = stablehlo.constant dense<100.0> : tensor<64x256xf32>
    %result   = stablehlo.divide %sum_red, %nsz : tensor<64x256xf32>
    return %result : tensor<64x256xf32>
  }
}
