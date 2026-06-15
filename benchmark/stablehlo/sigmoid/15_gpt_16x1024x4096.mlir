module @f_15_gpt_16x1024x4096 {
  func.func @f_15_gpt_16x1024x4096(%input: tensor<16x1024x4096xf32>) -> tensor<16x1024x4096xf32> {
    %result = stablehlo.logistic %input : tensor<16x1024x4096xf32>
    return %result : tensor<16x1024x4096xf32>
  }
}
