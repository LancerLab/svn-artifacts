module @f_13_dynamic_32x512xV1_32x512xV2_32x512xV1pV2 {
  func.func @f_13_dynamic_32x512xV1_32x512xV2_32x512xV1pV2(%in0: tensor<32x512x?xf32>, %in1: tensor<32x512x?xf32>) -> tensor<32x512x?xf32> {
    %result = stablehlo.concatenate %in0, %in1, dim = 2 : (tensor<32x512x?xf32>, tensor<32x512x?xf32>) -> tensor<32x512x?xf32>
    return %result : tensor<32x512x?xf32>
  }
}
