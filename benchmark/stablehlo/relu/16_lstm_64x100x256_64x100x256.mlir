module @f_16_lstm_64x100x256_64x100x256 {
  func.func @f_16_lstm_64x100x256_64x100x256(%input: tensor<64x100x256xf32>) -> tensor<64x100x256xf32> {
    %zero   = stablehlo.constant dense<0.0> : tensor<f32>
    %inf    = stablehlo.constant dense<3.4028235e+38> : tensor<f32>
    %result = stablehlo.clamp %zero, %input, %inf : (tensor<f32>, tensor<64x100x256xf32>, tensor<f32>) -> tensor<64x100x256xf32>
    return %result : tensor<64x100x256xf32>
  }
}
