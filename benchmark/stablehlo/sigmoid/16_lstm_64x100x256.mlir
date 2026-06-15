module @f_16_lstm_64x100x256 {
  func.func @f_16_lstm_64x100x256(%input: tensor<64x100x256xf32>) -> tensor<64x100x256xf32> {
    %result = stablehlo.logistic %input : tensor<64x100x256xf32>
    return %result : tensor<64x100x256xf32>
  }
}
