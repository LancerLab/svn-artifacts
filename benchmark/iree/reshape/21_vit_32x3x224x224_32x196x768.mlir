module {
  func.func @f_21_vit_32x3x224x224_32x196x768(%input: tensor<32x3x224x224xf32>) -> tensor<32x196x768xf32> {
    %flat = tensor.collapse_shape %input [[0, 1, 2, 3]] : tensor<32x3x224x224xf32> into tensor<4816896xf32>
    %out = tensor.expand_shape %flat [[0, 1, 2]] output_shape [32, 196, 768] : tensor<4816896xf32> into tensor<32x196x768xf32>
    return %out : tensor<32x196x768xf32>
  }
}
