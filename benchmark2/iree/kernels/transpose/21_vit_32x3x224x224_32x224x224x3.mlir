module {
  func.func @f_21_vit_32x3x224x224_32x224x224x3(%input: tensor<32x3x224x224xf32>) -> tensor<32x224x224x3xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<32x3x224x224xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<32x3x224x224xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<32x3x224x224xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<32x3x224x224xf32>
    %out = tensor.empty() : tensor<32x224x224x3xf32>
    %result = linalg.transpose ins(%input : tensor<32x3x224x224xf32>) outs(%out : tensor<32x224x224x3xf32>) permutation = [0, 2, 3, 1]
    return %result : tensor<32x224x224x3xf32>
  }
}
