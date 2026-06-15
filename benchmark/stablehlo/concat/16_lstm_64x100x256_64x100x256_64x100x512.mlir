module @f_16_lstm_64x100x256_64x100x256_64x100x512 {
  func.func @f_16_lstm_64x100x256_64x100x256_64x100x512(%in0: tensor<64x100x256xf32>, %in1: tensor<64x100x256xf32>) -> tensor<64x100x512xf32> {
    %result = stablehlo.concatenate %in0, %in1, dim = 2 : (tensor<64x100x256xf32>, tensor<64x100x256xf32>) -> tensor<64x100x512xf32>
    return %result : tensor<64x100x512xf32>
  }
}
