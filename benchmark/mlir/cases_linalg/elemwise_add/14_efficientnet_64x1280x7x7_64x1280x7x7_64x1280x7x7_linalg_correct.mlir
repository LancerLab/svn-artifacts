module {
  func.func @f_14_efficientnet_64x1280x7x7_64x1280x7x7_64x1280x7x7_correct(%a: tensor<64x1280x7x7xf32>, %b: tensor<64x1280x7x7xf32>) -> tensor<64x1280x7x7xf32> {
    %init = tensor.empty() : tensor<64x1280x7x7xf32>
    %r = linalg.add ins(%a, %b : tensor<64x1280x7x7xf32>, tensor<64x1280x7x7xf32>)
                    outs(%init : tensor<64x1280x7x7xf32>) -> tensor<64x1280x7x7xf32>
    return %r : tensor<64x1280x7x7xf32>
  }
}
