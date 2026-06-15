module @f_19_transformer_32x512x2048 {
  func.func @f_19_transformer_32x512x2048(%input: tensor<32x512x2048xf32>) -> tensor<32x512x2048xf32> {
    %result = stablehlo.logistic %input : tensor<32x512x2048xf32>
    return %result : tensor<32x512x2048xf32>
  }
}
