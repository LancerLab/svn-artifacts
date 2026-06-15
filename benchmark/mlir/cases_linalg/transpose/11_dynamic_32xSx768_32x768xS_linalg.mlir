module {
  func.func @f_11_dynamic_32xSx768_32x768xS_linalg(%input: tensor<32x?x768xf32>) -> tensor<32x768x?xf32> {
    %c1_2 = arith.constant 1 : index
    %d2 = tensor.dim %input, %c1_2 : tensor<32x?x768xf32>
    %init = tensor.empty(%d2) : tensor<32x768x?xf32>
    %r = linalg.transpose ins(%input : tensor<32x?x768xf32>) outs(%init : tensor<32x768x?xf32>) permutation = [0, 2, 1]
    return %r : tensor<32x768x?xf32>
  }
}
