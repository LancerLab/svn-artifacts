module {
  func.func @f_14_efficientnet_64x1280x7x7_64x7x7x1280(%input: tensor<64x1280x7x7xf32>) -> tensor<64x7x7x1280xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<64x1280x7x7xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<64x1280x7x7xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<64x1280x7x7xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<64x1280x7x7xf32>
    %out = tensor.empty() : tensor<64x7x7x1280xf32>
    %result = linalg.transpose ins(%input : tensor<64x1280x7x7xf32>) outs(%out : tensor<64x7x7x1280xf32>) permutation = [0, 2, 3, 1]
    return %result : tensor<64x7x7x1280xf32>
  }
}
