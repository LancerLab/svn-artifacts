module @f_9_dynamic_64x128xHxW {
  func.func @f_9_dynamic_64x128xHxW(%input: tensor<64x128x?x?xf32>) -> tensor<64x128x?x?xf32> {
    %result = stablehlo.logistic %input : tensor<64x128x?x?xf32>
    return %result : tensor<64x128x?x?xf32>
  }
}
