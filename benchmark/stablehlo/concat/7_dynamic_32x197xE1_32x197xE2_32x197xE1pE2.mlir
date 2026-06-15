module @f_7_dynamic_32x197xE1_32x197xE2_32x197xE1pE2 {
  func.func @f_7_dynamic_32x197xE1_32x197xE2_32x197xE1pE2(%in0: tensor<32x197x?xf32>, %in1: tensor<32x197x?xf32>) -> tensor<32x197x?xf32> {
    %result = stablehlo.concatenate %in0, %in1, dim = 2 : (tensor<32x197x?xf32>, tensor<32x197x?xf32>) -> tensor<32x197x?xf32>
    return %result : tensor<32x197x?xf32>
  }
}
