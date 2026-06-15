module @f_7_dynamic_32x197xD {
  func.func @f_7_dynamic_32x197xD(%input: tensor<32x197x?xf32>) -> tensor<32x197x?xf32> {
    %result = stablehlo.logistic %input : tensor<32x197x?xf32>
    return %result : tensor<32x197x?xf32>
  }
}
