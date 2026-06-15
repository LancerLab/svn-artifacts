module @f_15_gpt_16x1024x1024_16x16x1024x64 {
  func.func @f_15_gpt_16x1024x1024_16x16x1024x64(%input: tensor<16x1024x1024xf32>) -> tensor<16x16x1024x64xf32> {
    %result = stablehlo.reshape %input : (tensor<16x1024x1024xf32>) -> tensor<16x16x1024x64xf32>
    return %result : tensor<16x16x1024x64xf32>
  }
}
