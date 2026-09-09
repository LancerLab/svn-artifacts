module {
  func.func @f_16_lstm_64x100x256_100x64x256(%input: tensor<64x100x256xf32>) -> tensor<100x64x256xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<64x100x256xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<64x100x256xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<64x100x256xf32>
    %out = tensor.empty() : tensor<100x64x256xf32>
    %result = linalg.transpose ins(%input : tensor<64x100x256xf32>) outs(%out : tensor<100x64x256xf32>) permutation = [1, 0, 2]
    return %result : tensor<100x64x256xf32>
  }
}
