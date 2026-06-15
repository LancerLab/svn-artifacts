module @f_3_attention_32xNx512x64_32x512xNx64 {
  func.func @f_3_attention_32xNx512x64_32x512xNx64(%input: tensor<32x?x512x64xf32>) -> tensor<32x512x?x64xf32> {
    %result = stablehlo.transpose %input, dims = [0, 2, 1, 3] : (tensor<32x?x512x64xf32>) -> tensor<32x512x?x64xf32>
    return %result : tensor<32x512x?x64xf32>
  }
}
