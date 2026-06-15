module {
  func.func @f_13_dynamic_32x512xV_32xVx512_linalg(%input: tensor<32x512x?xf32>) -> tensor<32x?x512xf32> {
    %c2_1 = arith.constant 2 : index
    %d1 = tensor.dim %input, %c2_1 : tensor<32x512x?xf32>
    %init = tensor.empty(%d1) : tensor<32x?x512xf32>
    %r = linalg.transpose ins(%input : tensor<32x512x?xf32>) outs(%init : tensor<32x?x512xf32>) permutation = [0, 2, 1]
    return %r : tensor<32x?x512xf32>
  }
}
