module {
  func.func @f_2_cnn_128x128x28x28_128x100352(%input: tensor<128x128x28x28xf32>) -> tensor<128x100352xf32> {
    %out = tensor.collapse_shape %input [[0], [1, 2, 3]] : tensor<128x128x28x28xf32> into tensor<128x100352xf32>
    return %out : tensor<128x100352xf32>
  }
}
