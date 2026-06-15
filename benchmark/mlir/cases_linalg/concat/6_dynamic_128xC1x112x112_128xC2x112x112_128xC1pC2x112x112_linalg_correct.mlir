module {
  func.func @f_6_dynamic_128xC1x112x112_128xC2x112x112_128xC1pC2x112x112_correct(%in0: tensor<128x?x112x112xf32>, %in1: tensor<128x?x112x112xf32>) -> tensor<128x?x112x112xf32> {
    %r = tensor.concat dim(1) %in0, %in1 : (tensor<128x?x112x112xf32>, tensor<128x?x112x112xf32>) -> tensor<128x?x112x112xf32>
    return %r : tensor<128x?x112x112xf32>
  }
}
