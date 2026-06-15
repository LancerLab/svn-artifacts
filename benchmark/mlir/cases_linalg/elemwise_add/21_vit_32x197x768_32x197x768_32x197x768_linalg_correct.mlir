module {
  func.func @f_21_vit_32x197x768_32x197x768_32x197x768_correct(%a: tensor<32x197x768xf32>, %b: tensor<32x197x768xf32>) -> tensor<32x197x768xf32> {
    %init = tensor.empty() : tensor<32x197x768xf32>
    %r = linalg.add ins(%a, %b : tensor<32x197x768xf32>, tensor<32x197x768xf32>)
                    outs(%init : tensor<32x197x768xf32>) -> tensor<32x197x768xf32>
    return %r : tensor<32x197x768xf32>
  }
}
