module {
  func.func @f_15_gpt_16x16x1024x64_16x1024x16x64_linalg(%input: tensor<16x16x1024x64xf32>) -> tensor<16x1024x16x64xf32> {
    %init = tensor.empty() : tensor<16x1024x16x64xf32>
    %r = linalg.transpose ins(%input : tensor<16x16x1024x64xf32>) outs(%init : tensor<16x1024x16x64xf32>) permutation = [0, 2, 1, 3]
    return %r : tensor<16x1024x16x64xf32>
  }
}
