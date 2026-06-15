module {
  func.func @f_7_dynamic_32x197xD_32xDx197_linalg(%input: tensor<32x197x?xf32>) -> tensor<32x?x197xf32> {
    %c2_1 = arith.constant 2 : index
    %d1 = tensor.dim %input, %c2_1 : tensor<32x197x?xf32>
    %init = tensor.empty(%d1) : tensor<32x?x197xf32>
    %r = linalg.transpose ins(%input : tensor<32x197x?xf32>) outs(%init : tensor<32x?x197xf32>) permutation = [0, 2, 1]
    return %r : tensor<32x?x197xf32>
  }
}
