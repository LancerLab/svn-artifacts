module @f_7_dynamic_32x197xD_32xDx197 {
  func.func @f_7_dynamic_32x197xD_32xDx197(%input: tensor<32x197x?xf32>) -> tensor<32x?x197xf32> {
    %result = stablehlo.transpose %input, dims = [0, 2, 1] : (tensor<32x197x?xf32>) -> tensor<32x?x197xf32>
    return %result : tensor<32x?x197xf32>
  }
}
