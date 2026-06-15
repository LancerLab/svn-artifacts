module {
  func.func @f_8_dynamic_16x1024xD_16xDx1024_linalg(%input: tensor<16x1024x?xf32>) -> tensor<16x?x1024xf32> {
    %c2_1 = arith.constant 2 : index
    %d1 = tensor.dim %input, %c2_1 : tensor<16x1024x?xf32>
    %init = tensor.empty(%d1) : tensor<16x?x1024xf32>
    %r = linalg.transpose ins(%input : tensor<16x1024x?xf32>) outs(%init : tensor<16x?x1024xf32>) permutation = [0, 2, 1]
    return %r : tensor<16x?x1024xf32>
  }
}
