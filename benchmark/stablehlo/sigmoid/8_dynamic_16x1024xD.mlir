module @f_8_dynamic_16x1024xD {
  func.func @f_8_dynamic_16x1024xD(%input: tensor<16x1024x?xf32>) -> tensor<16x1024x?xf32> {
    %result = stablehlo.logistic %input : tensor<16x1024x?xf32>
    return %result : tensor<16x1024x?xf32>
  }
}
