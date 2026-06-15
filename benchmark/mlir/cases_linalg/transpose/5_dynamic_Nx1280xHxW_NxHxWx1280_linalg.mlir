module {
  func.func @f_5_dynamic_Nx1280xHxW_NxHxWx1280_linalg(%input: tensor<?x1280x?x?xf32>) -> tensor<?x?x?x1280xf32> {
    %c0_0 = arith.constant 0 : index
    %d0 = tensor.dim %input, %c0_0 : tensor<?x1280x?x?xf32>
    %c2_1 = arith.constant 2 : index
    %d1 = tensor.dim %input, %c2_1 : tensor<?x1280x?x?xf32>
    %c3_2 = arith.constant 3 : index
    %d2 = tensor.dim %input, %c3_2 : tensor<?x1280x?x?xf32>
    %init = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?x1280xf32>
    %r = linalg.transpose ins(%input : tensor<?x1280x?x?xf32>) outs(%init : tensor<?x?x?x1280xf32>) permutation = [0, 2, 3, 1]
    return %r : tensor<?x?x?x1280xf32>
  }
}
