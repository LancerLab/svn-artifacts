module {
  func.func @f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7_correct(%in0: tensor<64x1280x7x7xf32>, %in1: tensor<64x320x7x7xf32>) -> tensor<64x1600x7x7xf32> {
    %r = tensor.concat dim(1) %in0, %in1 : (tensor<64x1280x7x7xf32>, tensor<64x320x7x7xf32>) -> tensor<64x1600x7x7xf32>
    return %r : tensor<64x1600x7x7xf32>
  }
}
