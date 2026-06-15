module @f_4_attention_32xNx512x64_32xNx512x64_32xNx512x64 {
  func.func @f_4_attention_32xNx512x64_32xNx512x64_32xNx512x64(%input0: tensor<32x?x512x64xf32>, %input1: tensor<32x?x512x64xf32>) -> tensor<32x?x512x64xf32> {
    %result = stablehlo.add %input0, %input1 : tensor<32x?x512x64xf32>
    return %result : tensor<32x?x512x64xf32>
  }
}
