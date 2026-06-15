module @f_14_efficientnet_64x1280x7x7_64x7x7x1280 {
  func.func @f_14_efficientnet_64x1280x7x7_64x7x7x1280(%input: tensor<64x1280x7x7xf32>) -> tensor<64x7x7x1280xf32> {
    %result = stablehlo.transpose %input, dims = [0, 2, 3, 1] : (tensor<64x1280x7x7xf32>) -> tensor<64x7x7x1280xf32>
    return %result : tensor<64x7x7x1280xf32>
  }
}
