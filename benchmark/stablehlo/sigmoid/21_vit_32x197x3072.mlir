module @f_21_vit_32x197x3072 {
  func.func @f_21_vit_32x197x3072(%input: tensor<32x197x3072xf32>) -> tensor<32x197x3072xf32> {
    %result = stablehlo.logistic %input : tensor<32x197x3072xf32>
    return %result : tensor<32x197x3072xf32>
  }
}
