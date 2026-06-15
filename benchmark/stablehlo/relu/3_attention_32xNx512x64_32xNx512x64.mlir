module @f_3_attention_32xNx512x64_32xNx512x64 {
  func.func @f_3_attention_32xNx512x64_32xNx512x64(%input: tensor<32x?x512x64xf32>) -> tensor<32x?x512x64xf32> {
    %zero   = stablehlo.constant dense<0.0> : tensor<f32>
    %inf    = stablehlo.constant dense<3.4028235e+38> : tensor<f32>
    %result = stablehlo.clamp %zero, %input, %inf : (tensor<f32>, tensor<32x?x512x64xf32>, tensor<f32>) -> tensor<32x?x512x64xf32>
    return %result : tensor<32x?x512x64xf32>
  }
}
