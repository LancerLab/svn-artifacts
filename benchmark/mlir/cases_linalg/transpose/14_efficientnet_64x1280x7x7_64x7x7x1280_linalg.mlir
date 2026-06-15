module {
  func.func @f_14_efficientnet_64x1280x7x7_64x7x7x1280_linalg(%input: tensor<64x1280x7x7xf32>) -> tensor<64x7x7x1280xf32> {
    %init = tensor.empty() : tensor<64x7x7x1280xf32>
    %r = linalg.transpose ins(%input : tensor<64x1280x7x7xf32>) outs(%init : tensor<64x7x7x1280xf32>) permutation = [0, 2, 3, 1]
    return %r : tensor<64x7x7x1280xf32>
  }
}
