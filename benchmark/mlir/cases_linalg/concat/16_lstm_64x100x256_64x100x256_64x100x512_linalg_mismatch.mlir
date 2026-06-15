module {
  func.func @f_16_lstm_64x100x256_64x100x256_64x100x512_mismatch(%in0: tensor<64x100x256xf32>, %in1: tensor<63x100x256xf32>) -> tensor<64x100x512xf32> {
    %r = tensor.concat dim(2) %in0, %in1 : (tensor<64x100x256xf32>, tensor<63x100x256xf32>) -> tensor<64x100x512xf32>
    return %r : tensor<64x100x512xf32>
  }
}
