module {
  func.func @f_10_dynamic_16x512xHxW_16xHxWx512(%input: tensor<16x512x?x?xf32>) -> tensor<16x?x?x512xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<16x512x?x?xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<16x512x?x?xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<16x512x?x?xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<16x512x?x?xf32>
    %out = tensor.empty(%input_d2, %input_d3) : tensor<16x?x?x512xf32>
    %result = linalg.transpose ins(%input : tensor<16x512x?x?xf32>) outs(%out : tensor<16x?x?x512xf32>) permutation = [0, 2, 3, 1]
    return %result : tensor<16x?x?x512xf32>
  }
}
