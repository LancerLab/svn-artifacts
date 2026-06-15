module @f_14_efficientnet_64x1280x7x7_64x62720 {
  func.func @f_14_efficientnet_64x1280x7x7_64x62720(%input: tensor<64x1280x7x7xf32>) -> tensor<64x62720xf32> {
    %result = stablehlo.reshape %input : (tensor<64x1280x7x7xf32>) -> tensor<64x62720xf32>
    return %result : tensor<64x62720xf32>
  }
}
