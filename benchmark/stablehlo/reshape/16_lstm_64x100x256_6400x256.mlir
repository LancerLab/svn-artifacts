module @f_16_lstm_64x100x256_6400x256 {
  func.func @f_16_lstm_64x100x256_6400x256(%input: tensor<64x100x256xf32>) -> tensor<6400x256xf32> {
    %result = stablehlo.reshape %input : (tensor<64x100x256xf32>) -> tensor<6400x256xf32>
    return %result : tensor<6400x256xf32>
  }
}
