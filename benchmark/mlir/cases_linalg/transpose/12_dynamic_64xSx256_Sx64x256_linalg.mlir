module {
  func.func @f_12_dynamic_64xSx256_Sx64x256_linalg(%input: tensor<64x?x256xf32>) -> tensor<?x64x256xf32> {
    %c1_0 = arith.constant 1 : index
    %d0 = tensor.dim %input, %c1_0 : tensor<64x?x256xf32>
    %init = tensor.empty(%d0) : tensor<?x64x256xf32>
    %r = linalg.transpose ins(%input : tensor<64x?x256xf32>) outs(%init : tensor<?x64x256xf32>) permutation = [1, 0, 2]
    return %r : tensor<?x64x256xf32>
  }
}
