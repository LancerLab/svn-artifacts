module {
  func.func @f_17_mobilenet_128x96x112x112_128x32x112x112_128x128x112x112_correct(%in0: tensor<128x96x112x112xf32>, %in1: tensor<128x32x112x112xf32>) -> tensor<128x128x112x112xf32> {
    %r = tensor.concat dim(1) %in0, %in1 : (tensor<128x96x112x112xf32>, tensor<128x32x112x112xf32>) -> tensor<128x128x112x112xf32>
    return %r : tensor<128x128x112x112xf32>
  }
}
