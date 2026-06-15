module {
  func.func @f_18_resnet_64x256x56x56_64x256x56x56_64x512x56x56_correct(%in0: tensor<64x256x56x56xf32>, %in1: tensor<64x256x56x56xf32>) -> tensor<64x512x56x56xf32> {
    %r = tensor.concat dim(1) %in0, %in1 : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x512x56x56xf32>
    return %r : tensor<64x512x56x56xf32>
  }
}
