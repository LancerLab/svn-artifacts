module {
  func.func @f_16_lstm_64x100x256_100x64x256_linalg(%input: tensor<64x100x256xf32>) -> tensor<100x64x256xf32> {
    %init = tensor.empty() : tensor<100x64x256xf32>
    %r = linalg.transpose ins(%input : tensor<64x100x256xf32>) outs(%init : tensor<100x64x256xf32>) permutation = [1, 0, 2]
    return %r : tensor<100x64x256xf32>
  }
}
