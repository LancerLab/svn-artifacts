module @f_10_dynamic_128x1280_1280xN_128xN {
  func.func @f_10_dynamic_128x1280_1280xN_128xN(%input0: tensor<128x1280xf32>, %input1: tensor<1280x?xf32>) -> tensor<128x?xf32> {
    %result = stablehlo.dot %input0, %input1 : (tensor<128x1280xf32>, tensor<1280x?xf32>) -> tensor<128x?xf32>
    return %result : tensor<128x?xf32>
  }
}
