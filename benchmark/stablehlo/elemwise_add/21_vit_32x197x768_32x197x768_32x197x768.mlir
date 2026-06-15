module @f_21_vit_32x197x768_32x197x768_32x197x768 {
  func.func @f_21_vit_32x197x768_32x197x768_32x197x768(%input0: tensor<32x197x768xf32>, %input1: tensor<32x197x768xf32>) -> tensor<32x197x768xf32> {
    %result = stablehlo.add %input0, %input1 : tensor<32x197x768xf32>
    return %result : tensor<32x197x768xf32>
  }
}
