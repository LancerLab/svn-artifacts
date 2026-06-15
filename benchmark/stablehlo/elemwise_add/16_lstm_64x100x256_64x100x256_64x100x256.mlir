module @f_16_lstm_64x100x256_64x100x256_64x100x256 {
  func.func @f_16_lstm_64x100x256_64x100x256_64x100x256(%input0: tensor<64x100x256xf32>, %input1: tensor<64x100x256xf32>) -> tensor<64x100x256xf32> {
    %result = stablehlo.add %input0, %input1 : tensor<64x100x256xf32>
    return %result : tensor<64x100x256xf32>
  }
}
