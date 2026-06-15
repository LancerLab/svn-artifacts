module @f_19_transformer_32x512x2048_32x1048576 {
  func.func @f_19_transformer_32x512x2048_32x1048576(%input: tensor<32x512x2048xf32>) -> tensor<32x1048576xf32> {
    %result = stablehlo.reshape %input : (tensor<32x512x2048xf32>) -> tensor<32x1048576xf32>
    return %result : tensor<32x1048576xf32>
  }
}
