module @f_21_vit_32x196x768_32x1x768_32x197x768 {
  func.func @f_21_vit_32x196x768_32x1x768_32x197x768(%in0: tensor<32x196x768xf32>, %in1: tensor<32x1x768xf32>) -> tensor<32x197x768xf32> {
    %result = stablehlo.concatenate %in0, %in1, dim = 1 : (tensor<32x196x768xf32>, tensor<32x1x768xf32>) -> tensor<32x197x768xf32>
    return %result : tensor<32x197x768xf32>
  }
}
