module {
  func.func @f_13_dynamic_32x512xV_32x512V(%input: tensor<32x512x?xf32>) -> tensor<32x?xf32> {
    %out = tensor.collapse_shape %input [[0], [1, 2]] : tensor<32x512x?xf32> into tensor<32x?xf32>
    return %out : tensor<32x?xf32>
  }
}
