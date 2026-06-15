module @f_15_gpt_16x512x1536_16x512x1536_16x512x3072 {
  func.func @f_15_gpt_16x512x1536_16x512x1536_16x512x3072(%in0: tensor<16x512x1536xf32>, %in1: tensor<16x512x1536xf32>) -> tensor<16x512x3072xf32> {
    %result = stablehlo.concatenate %in0, %in1, dim = 2 : (tensor<16x512x1536xf32>, tensor<16x512x1536xf32>) -> tensor<16x512x3072xf32>
    return %result : tensor<16x512x3072xf32>
  }
}
