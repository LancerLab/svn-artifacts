module @f_13_dynamic_32x512xV_32x512xV_32x512xV {
  func.func @f_13_dynamic_32x512xV_32x512xV_32x512xV(%input0: tensor<32x512x?xf32>, %input1: tensor<32x512x?xf32>) -> tensor<32x512x?xf32> {
    %result = stablehlo.add %input0, %input1 : tensor<32x512x?xf32>
    return %result : tensor<32x512x?xf32>
  }
}
