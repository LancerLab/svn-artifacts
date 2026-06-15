module @f_8_dynamic_16x1024xD_16x1024xD_16x1024xD {
  func.func @f_8_dynamic_16x1024xD_16x1024xD_16x1024xD(%input0: tensor<16x1024x?xf32>, %input1: tensor<16x1024x?xf32>) -> tensor<16x1024x?xf32> {
    %result = stablehlo.add %input0, %input1 : tensor<16x1024x?xf32>
    return %result : tensor<16x1024x?xf32>
  }
}
