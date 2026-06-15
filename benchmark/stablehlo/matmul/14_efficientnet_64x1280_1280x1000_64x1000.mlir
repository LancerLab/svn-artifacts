module @f_14_efficientnet_64x1280_1280x1000_64x1000 {
  func.func @f_14_efficientnet_64x1280_1280x1000_64x1000(%input0: tensor<64x1280xf32>, %input1: tensor<1280x1000xf32>) -> tensor<64x1000xf32> {
    %result = stablehlo.dot %input0, %input1 : (tensor<64x1280xf32>, tensor<1280x1000xf32>) -> tensor<64x1000xf32>
    return %result : tensor<64x1000xf32>
  }
}
