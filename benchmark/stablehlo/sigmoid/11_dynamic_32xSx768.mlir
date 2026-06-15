module @f_11_dynamic_32xSx768 {
  func.func @f_11_dynamic_32xSx768(%input: tensor<32x?x768xf32>) -> tensor<32x?x768xf32> {
    %result = stablehlo.logistic %input : tensor<32x?x768xf32>
    return %result : tensor<32x?x768xf32>
  }
}
