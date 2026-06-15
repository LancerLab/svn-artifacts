module {
  func.func @f_10_dynamic_16x512xHxW_16xHxWx512_linalg(%input: tensor<16x512x?x?xf32>) -> tensor<16x?x?x512xf32> {
    %c2_1 = arith.constant 2 : index
    %d1 = tensor.dim %input, %c2_1 : tensor<16x512x?x?xf32>
    %c3_2 = arith.constant 3 : index
    %d2 = tensor.dim %input, %c3_2 : tensor<16x512x?x?xf32>
    %init = tensor.empty(%d1, %d2) : tensor<16x?x?x512xf32>
    %r = linalg.transpose ins(%input : tensor<16x512x?x?xf32>) outs(%init : tensor<16x?x?x512xf32>) permutation = [0, 2, 3, 1]
    return %r : tensor<16x?x?x512xf32>
  }
}
