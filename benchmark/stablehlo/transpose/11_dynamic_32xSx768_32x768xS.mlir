module @f_11_dynamic_32xSx768_32x768xS {
  func.func @f_11_dynamic_32xSx768_32x768xS(%input: tensor<32x?x768xf32>) -> tensor<32x768x?xf32> {
    %result = stablehlo.transpose %input, dims = [0, 2, 1] : (tensor<32x?x768xf32>) -> tensor<32x768x?xf32>
    return %result : tensor<32x768x?xf32>
  }
}
