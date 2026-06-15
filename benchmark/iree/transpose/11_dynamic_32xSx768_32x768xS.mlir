module {
  func.func @f_11_dynamic_32xSx768_32x768xS(%input: tensor<32x?x768xf32>) -> tensor<32x768x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<32x?x768xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<32x?x768xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<32x?x768xf32>
    %out = tensor.empty(%input_d1) : tensor<32x768x?xf32>
    %result = linalg.transpose ins(%input : tensor<32x?x768xf32>) outs(%out : tensor<32x768x?xf32>) permutation = [0, 2, 1]
    return %result : tensor<32x768x?xf32>
  }
}
