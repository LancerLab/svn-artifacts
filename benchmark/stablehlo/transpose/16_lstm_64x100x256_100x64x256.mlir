module @f_16_lstm_64x100x256_100x64x256 {
  func.func @f_16_lstm_64x100x256_100x64x256(%input: tensor<64x100x256xf32>) -> tensor<100x64x256xf32> {
    %result = stablehlo.transpose %input, dims = [1, 0, 2] : (tensor<64x100x256xf32>) -> tensor<100x64x256xf32>
    return %result : tensor<100x64x256xf32>
  }
}
