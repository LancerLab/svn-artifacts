module @f_6_dynamic_128xCx112x112_128x112x112xC {
  func.func @f_6_dynamic_128xCx112x112_128x112x112xC(%input: tensor<128x?x112x112xf32>) -> tensor<128x112x112x?xf32> {
    %result = stablehlo.transpose %input, dims = [0, 2, 3, 1] : (tensor<128x?x112x112xf32>) -> tensor<128x112x112x?xf32>
    return %result : tensor<128x112x112x?xf32>
  }
}
