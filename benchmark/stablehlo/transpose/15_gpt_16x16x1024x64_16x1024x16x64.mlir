module @f_15_gpt_16x16x1024x64_16x1024x16x64 {
  func.func @f_15_gpt_16x16x1024x64_16x1024x16x64(%input: tensor<16x16x1024x64xf32>) -> tensor<16x1024x16x64xf32> {
    %result = stablehlo.transpose %input, dims = [0, 2, 1, 3] : (tensor<16x16x1024x64xf32>) -> tensor<16x1024x16x64xf32>
    return %result : tensor<16x1024x16x64xf32>
  }
}
