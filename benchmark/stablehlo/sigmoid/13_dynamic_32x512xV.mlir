module @f_13_dynamic_32x512xV {
  func.func @f_13_dynamic_32x512xV(%input: tensor<32x512x?xf32>) -> tensor<32x512x?xf32> {
    %result = stablehlo.logistic %input : tensor<32x512x?xf32>
    return %result : tensor<32x512x?xf32>
  }
}
