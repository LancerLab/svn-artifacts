module @f_8_dynamic_16x1024xD1_16x1024xD2_16x1024xD1pD2 {
  func.func @f_8_dynamic_16x1024xD1_16x1024xD2_16x1024xD1pD2(%in0: tensor<16x1024x?xf32>, %in1: tensor<16x1024x?xf32>) -> tensor<16x1024x?xf32> {
    %result = stablehlo.concatenate %in0, %in1, dim = 2 : (tensor<16x1024x?xf32>, tensor<16x1024x?xf32>) -> tensor<16x1024x?xf32>
    return %result : tensor<16x1024x?xf32>
  }
}
