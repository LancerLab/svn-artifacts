module @f_17_mobilenet_128x96x112x112_128x96x112x112_128x96x112x112 {
  func.func @f_17_mobilenet_128x96x112x112_128x96x112x112_128x96x112x112(%input0: tensor<128x96x112x112xf32>, %input1: tensor<128x96x112x112xf32>) -> tensor<128x96x112x112xf32> {
    %result = stablehlo.add %input0, %input1 : tensor<128x96x112x112xf32>
    return %result : tensor<128x96x112x112xf32>
  }
}
