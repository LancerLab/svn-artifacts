module @f_10_dynamic_16x512xHxW {
  func.func @f_10_dynamic_16x512xHxW(%input: tensor<16x512x?x?xf32>) -> tensor<16x512x?x?xf32> {
    %result = stablehlo.logistic %input : tensor<16x512x?x?xf32>
    return %result : tensor<16x512x?x?xf32>
  }
}
