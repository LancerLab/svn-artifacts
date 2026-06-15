module @f_17_mobilenet_128xx96x112x112_128x112x112x96 {
  func.func @f_17_mobilenet_128xx96x112x112_128x112x112x96(%input: tensor<128x96x112x112xf32>) -> tensor<128x112x112x96xf32> {
    %result = stablehlo.transpose %input, dims = [0, 2, 3, 1] : (tensor<128x96x112x112xf32>) -> tensor<128x112x112x96xf32>
    return %result : tensor<128x112x112x96xf32>
  }
}
