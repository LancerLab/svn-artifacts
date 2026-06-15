module @f_3_attention_32xNx512x64 {
  func.func @f_3_attention_32xNx512x64(%input: tensor<32x?x512x64xf32>) -> tensor<32x?x512x64xf32> {
    %result = stablehlo.logistic %input : tensor<32x?x512x64xf32>
    return %result : tensor<32x?x512x64xf32>
  }
}
