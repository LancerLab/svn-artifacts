module {
  func.func @f_21_vit_32x196x768_32x1x768_32x197x768_correct(%in0: tensor<32x196x768xf32>, %in1: tensor<32x1x768xf32>) -> tensor<32x197x768xf32> {
    %r = tensor.concat dim(1) %in0, %in1 : (tensor<32x196x768xf32>, tensor<32x1x768xf32>) -> tensor<32x197x768xf32>
    return %r : tensor<32x197x768xf32>
  }
}
