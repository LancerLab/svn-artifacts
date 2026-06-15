module {
  func.func @f_19_transformer_32x512x2048_32x1048576(%input: tensor<32x512x2048xf32>) -> tensor<32x1048576xf32> {
    %out = tensor.collapse_shape %input [[0], [1, 2]] : tensor<32x512x2048xf32> into tensor<32x1048576xf32>
    return %out : tensor<32x1048576xf32>
  }
}
