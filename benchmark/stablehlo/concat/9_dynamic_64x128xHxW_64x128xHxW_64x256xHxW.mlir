module @f_9_dynamic_64x128xHxW_64x128xHxW_64x256xHxW {
  func.func @f_9_dynamic_64x128xHxW_64x128xHxW_64x256xHxW(%in0: tensor<64x128x?x?xf32>, %in1: tensor<64x128x?x?xf32>) -> tensor<64x256x?x?xf32> {
    %result = stablehlo.concatenate %in0, %in1, dim = 1 : (tensor<64x128x?x?xf32>, tensor<64x128x?x?xf32>) -> tensor<64x256x?x?xf32>
    return %result : tensor<64x256x?x?xf32>
  }
}
