module {
  func.func @f_19_transformer_32x512x2048_32x2048x512(%input: tensor<32x512x2048xf32>) -> tensor<32x2048x512xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<32x512x2048xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<32x512x2048xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<32x512x2048xf32>
    %out = tensor.empty() : tensor<32x2048x512xf32>
    %result = linalg.transpose ins(%input : tensor<32x512x2048xf32>) outs(%out : tensor<32x2048x512xf32>) permutation = [0, 2, 1]
    return %result : tensor<32x2048x512xf32>
  }
}
