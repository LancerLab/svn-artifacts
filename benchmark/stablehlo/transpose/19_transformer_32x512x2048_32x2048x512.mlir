module @f_19_transformer_32x512x2048_32x2048x512 {
  func.func @f_19_transformer_32x512x2048_32x2048x512(%input: tensor<32x512x2048xf32>) -> tensor<32x2048x512xf32> {
    %result = stablehlo.transpose %input, dims = [0, 2, 1] : (tensor<32x512x2048xf32>) -> tensor<32x2048x512xf32>
    return %result : tensor<32x2048x512xf32>
  }
}
