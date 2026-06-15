module @f_1_bert_32x512x768 {
  func.func @f_1_bert_32x512x768(%input: tensor<32x512x768xf32>) -> tensor<32x512x768xf32> {
    %result = stablehlo.logistic %input : tensor<32x512x768xf32>
    return %result : tensor<32x512x768xf32>
  }
}
