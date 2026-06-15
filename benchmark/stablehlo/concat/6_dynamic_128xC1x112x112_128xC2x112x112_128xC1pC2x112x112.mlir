module @f_6_dynamic_128xC1x112x112_128xC2x112x112_128xC1pC2x112x112 {
  func.func @f_6_dynamic_128xC1x112x112_128xC2x112x112_128xC1pC2x112x112(%in0: tensor<128x?x112x112xf32>, %in1: tensor<128x?x112x112xf32>) -> tensor<128x?x112x112xf32> {
    %result = stablehlo.concatenate %in0, %in1, dim = 1 : (tensor<128x?x112x112xf32>, tensor<128x?x112x112xf32>) -> tensor<128x?x112x112xf32>
    return %result : tensor<128x?x112x112xf32>
  }
}
