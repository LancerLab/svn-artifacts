module {
  func.func @f_18_resnet_64x512x7x7_64x25088(%input: tensor<64x512x7x7xf32>) -> tensor<64x25088xf32> {
    %out = tensor.collapse_shape %input [[0], [1, 2, 3]] : tensor<64x512x7x7xf32> into tensor<64x25088xf32>
    return %out : tensor<64x25088xf32>
  }
}
