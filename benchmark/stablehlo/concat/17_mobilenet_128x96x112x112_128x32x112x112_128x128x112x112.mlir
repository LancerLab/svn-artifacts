module @f_17_mobilenet_128x96x112x112_128x32x112x112_128x128x112x112 {
  func.func @f_17_mobilenet_128x96x112x112_128x32x112x112_128x128x112x112(%in0: tensor<128x96x112x112xf32>, %in1: tensor<128x32x112x112xf32>) -> tensor<128x128x112x112xf32> {
    %result = stablehlo.concatenate %in0, %in1, dim = 1 : (tensor<128x96x112x112xf32>, tensor<128x32x112x112xf32>) -> tensor<128x128x112x112xf32>
    return %result : tensor<128x128x112x112xf32>
  }
}
