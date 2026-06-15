module @f_14_efficientnet_64x1280x7x7 {
  func.func @f_14_efficientnet_64x1280x7x7(%input: tensor<64x1280x7x7xf32>) -> tensor<64x1280x7x7xf32> {
    %result = stablehlo.logistic %input : tensor<64x1280x7x7xf32>
    return %result : tensor<64x1280x7x7xf32>
  }
}
