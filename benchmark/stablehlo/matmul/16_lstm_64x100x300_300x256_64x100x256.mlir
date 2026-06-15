module @f_16_lstm_64x100x300_300x256_64x100x256 {
  func.func @f_16_lstm_64x100x300_300x256_64x100x256(%input0: tensor<64x100x300xf32>, %input1: tensor<300x256xf32>) -> tensor<64x100x256xf32> {
    %result = stablehlo.dot_general %input0, %input1,
        batching_dims = [] x [],
        contracting_dims = [2] x [0] : (tensor<64x100x300xf32>, tensor<300x256xf32>) -> tensor<64x100x256xf32>
    return %result : tensor<64x100x256xf32>
  }
}
