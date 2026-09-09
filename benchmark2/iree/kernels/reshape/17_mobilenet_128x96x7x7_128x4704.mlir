module {
  func.func @f_17_mobilenet_128x96x7x7_128x4704(%input: tensor<128x96x7x7xf32>) -> tensor<128x4704xf32> {
    %out = tensor.collapse_shape %input [[0], [1, 2, 3]] : tensor<128x96x7x7xf32> into tensor<128x4704xf32>
    return %out : tensor<128x4704xf32>
  }
}
