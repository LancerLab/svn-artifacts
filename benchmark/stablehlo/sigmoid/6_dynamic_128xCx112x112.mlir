module @f_6_dynamic_128xCx112x112 {
  func.func @f_6_dynamic_128xCx112x112(%input: tensor<128x?x112x112xf32>) -> tensor<128x?x112x112xf32> {
    %result = stablehlo.logistic %input : tensor<128x?x112x112xf32>
    return %result : tensor<128x?x112x112xf32>
  }
}
