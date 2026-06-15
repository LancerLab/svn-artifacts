module @f_11_dynamic_32xSx768_32xSx768_32xSx768 {
  func.func @f_11_dynamic_32xSx768_32xSx768_32xSx768(%input0: tensor<32x?x768xf32>, %input1: tensor<32x?x768xf32>) -> tensor<32x?x768xf32> {
    %result = stablehlo.add %input0, %input1 : tensor<32x?x768xf32>
    return %result : tensor<32x?x768xf32>
  }
}
