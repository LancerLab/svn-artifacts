module {
  func.func @f_14_efficientnet_64x1280x7x7_64x62720(%input: tensor<64x1280x7x7xf32>) -> tensor<64x62720xf32> {
    %out = tensor.collapse_shape %input [[0], [1, 2, 3]] : tensor<64x1280x7x7xf32> into tensor<64x62720xf32>
    return %out : tensor<64x62720xf32>
  }
}
