module @f_16_lstm_64x100x256_256_256_64x100x256 {
  func.func @f_16_lstm_64x100x256_256_256_64x100x256(%input: tensor<64x100x256xf32>, %gamma: tensor<256xf32>, %beta: tensor<256xf32>) -> tensor<64x100x256xf32> {
    %mean    = stablehlo.constant dense<0.0> : tensor<256xf32>
    %var     = stablehlo.constant dense<1.0> : tensor<256xf32>
    %result  = "stablehlo.batch_norm_inference"(%input, %gamma, %beta, %mean, %var)
        {epsilon = 1.000000e-05 : f32, feature_index = 2 : i64}
        : (tensor<64x100x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<64x100x256xf32>
    return %result : tensor<64x100x256xf32>
  }
}
