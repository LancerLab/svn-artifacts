module {
  func.func @f_9_dynamic_64x128xHxW_64x128xHxW_64x256xHxW_correct(%in0: tensor<64x128x?x?xf32>, %in1: tensor<64x128x?x?xf32>) -> tensor<64x256x?x?xf32> {
    %r = tensor.concat dim(1) %in0, %in1 : (tensor<64x128x?x?xf32>, tensor<64x128x?x?xf32>) -> tensor<64x256x?x?xf32>
    return %r : tensor<64x256x?x?xf32>
  }
}
