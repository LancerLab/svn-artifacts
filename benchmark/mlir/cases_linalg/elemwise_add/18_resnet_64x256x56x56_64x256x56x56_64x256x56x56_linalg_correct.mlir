module {
  func.func @f_18_resnet_64x256x56x56_64x256x56x56_64x256x56x56_correct(%a: tensor<64x256x56x56xf32>, %b: tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xf32> {
    %init = tensor.empty() : tensor<64x256x56x56xf32>
    %r = linalg.add ins(%a, %b : tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>)
                    outs(%init : tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xf32>
    return %r : tensor<64x256x56x56xf32>
  }
}
