module {
  func.func @f_15_gpt_16x512x1536_16x512x1536_16x512x3072_mismatch(%in0: tensor<16x512x1536xf32>, %in1: tensor<15x512x1536xf32>) -> tensor<16x512x3072xf32> {
    %r = tensor.concat dim(2) %in0, %in1 : (tensor<16x512x1536xf32>, tensor<15x512x1536xf32>) -> tensor<16x512x3072xf32>
    return %r : tensor<16x512x3072xf32>
  }
}
