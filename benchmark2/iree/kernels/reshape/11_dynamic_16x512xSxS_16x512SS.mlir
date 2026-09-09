module {
  func.func @f_11_dynamic_16x512xSxS_16x512SS(%input: tensor<16x512x?x?xf32>) -> tensor<16x?xf32> {
    %out = tensor.collapse_shape %input [[0], [1, 2, 3]] : tensor<16x512x?x?xf32> into tensor<16x?xf32>
    return %out : tensor<16x?xf32>
  }
}
