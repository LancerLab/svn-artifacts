module {
  func.func @f_3_attention_32xNx512x64_32x512xNx64(%input: tensor<32x?x512x64xf32>) -> tensor<32x512x?x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<32x?x512x64xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<32x?x512x64xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<32x?x512x64xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<32x?x512x64xf32>
    %out = tensor.empty(%input_d1) : tensor<32x512x?x64xf32>
    %result = linalg.transpose ins(%input : tensor<32x?x512x64xf32>) outs(%out : tensor<32x512x?x64xf32>) permutation = [0, 2, 1, 3]
    return %result : tensor<32x512x?x64xf32>
  }
}
