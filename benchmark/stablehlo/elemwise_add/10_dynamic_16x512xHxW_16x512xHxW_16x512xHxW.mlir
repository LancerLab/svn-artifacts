module @f_10_dynamic_16x512xHxW_16x512xHxW_16x512xHxW {
  func.func @f_10_dynamic_16x512xHxW_16x512xHxW_16x512xHxW(%input0: tensor<16x512x?x?xf32>, %input1: tensor<16x512x?x?xf32>) -> tensor<16x512x?x?xf32> {
    %result = stablehlo.add %input0, %input1 : tensor<16x512x?x?xf32>
    return %result : tensor<16x512x?x?xf32>
  }
}
