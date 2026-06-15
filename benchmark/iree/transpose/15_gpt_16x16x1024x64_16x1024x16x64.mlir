module {
  func.func @f_15_gpt_16x16x1024x64_16x1024x16x64(%input: tensor<16x16x1024x64xf32>) -> tensor<16x1024x16x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<16x16x1024x64xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<16x16x1024x64xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<16x16x1024x64xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<16x16x1024x64xf32>
    %out = tensor.empty() : tensor<16x1024x16x64xf32>
    %result = linalg.transpose ins(%input : tensor<16x16x1024x64xf32>) outs(%out : tensor<16x1024x16x64xf32>) permutation = [0, 2, 1, 3]
    return %result : tensor<16x1024x16x64xf32>
  }
}
