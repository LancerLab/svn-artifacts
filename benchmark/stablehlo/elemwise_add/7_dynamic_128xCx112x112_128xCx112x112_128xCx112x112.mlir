module @f_7_dynamic_128xCx112x112_128xCx112x112_128xCx112x112 {
  func.func @f_7_dynamic_128xCx112x112_128xCx112x112_128xCx112x112(%input0: tensor<128x?x112x112xf32>, %input1: tensor<128x?x112x112xf32>) -> tensor<128x?x112x112xf32> {
    %result = stablehlo.add %input0, %input1 : tensor<128x?x112x112xf32>
    return %result : tensor<128x?x112x112xf32>
  }
}
