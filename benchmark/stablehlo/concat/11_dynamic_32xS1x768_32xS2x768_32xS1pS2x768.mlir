module @f_11_dynamic_32xS1x768_32xS2x768_32xS1pS2x768 {
  func.func @f_11_dynamic_32xS1x768_32xS2x768_32xS1pS2x768(%in0: tensor<32x?x768xf32>, %in1: tensor<32x?x768xf32>) -> tensor<32x?x768xf32> {
    %result = stablehlo.concatenate %in0, %in1, dim = 1 : (tensor<32x?x768xf32>, tensor<32x?x768xf32>) -> tensor<32x?x768xf32>
    return %result : tensor<32x?x768xf32>
  }
}
