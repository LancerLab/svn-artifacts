module @f_1_bert_32x512x768_32x512x12x64 {
  func.func @f_1_bert_32x512x768_32x512x12x64(%input: tensor<32x512x768xf32>) -> tensor<32x512x12x64xf32> {
    %result = stablehlo.reshape %input : (tensor<32x512x768xf32>) -> tensor<32x512x12x64xf32>
    return %result : tensor<32x512x12x64xf32>
  }
}
