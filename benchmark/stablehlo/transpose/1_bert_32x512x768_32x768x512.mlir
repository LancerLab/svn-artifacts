module @f_1_bert_32x512x768_32x768x512 {
  func.func @f_1_bert_32x512x768_32x768x512(%input: tensor<32x512x768xf32>) -> tensor<32x768x512xf32> {
    %result = stablehlo.transpose %input, dims = [0, 2, 1] : (tensor<32x512x768xf32>) -> tensor<32x768x512xf32>
    return %result : tensor<32x768x512xf32>
  }
}
