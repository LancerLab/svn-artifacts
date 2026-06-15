module {
  func.func @f_18_resnet_64x256x56x56_64x56x56x256_linalg(%input: tensor<64x256x56x56xf32>) -> tensor<64x56x56x256xf32> {
    %init = tensor.empty() : tensor<64x56x56x256xf32>
    %r = linalg.transpose ins(%input : tensor<64x256x56x56xf32>) outs(%init : tensor<64x56x56x256xf32>) permutation = [0, 2, 3, 1]
    return %r : tensor<64x56x56x256xf32>
  }
}
