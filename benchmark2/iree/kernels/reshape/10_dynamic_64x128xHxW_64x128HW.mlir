module {
  func.func @f_10_dynamic_64x128xHxW_64x128HW(%input: tensor<64x128x?x?xf32>) -> tensor<64x?xf32> {
    %out = tensor.collapse_shape %input [[0], [1, 2, 3]] : tensor<64x128x?x?xf32> into tensor<64x?xf32>
    return %out : tensor<64x?xf32>
  }
}
