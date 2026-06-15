module @f_9_dynamic_64x128xWxH_64x128xWxH_64x128xWxH {
  func.func @f_9_dynamic_64x128xWxH_64x128xWxH_64x128xWxH(%input0: tensor<64x128x?x?xf32>, %input1: tensor<64x128x?x?xf32>) -> tensor<64x128x?x?xf32> {
    %result = stablehlo.add %input0, %input1 : tensor<64x128x?x?xf32>
    return %result : tensor<64x128x?x?xf32>
  }
}
