module @f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7 {
  func.func @f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7(%in0: tensor<64x1280x7x7xf32>, %in1: tensor<64x320x7x7xf32>) -> tensor<64x1600x7x7xf32> {
    %result = stablehlo.concatenate %in0, %in1, dim = 1 : (tensor<64x1280x7x7xf32>, tensor<64x320x7x7xf32>) -> tensor<64x1600x7x7xf32>
    return %result : tensor<64x1600x7x7xf32>
  }
}
