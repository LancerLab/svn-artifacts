module @f_18_resnet_64x256x56x56_64x256x56x56_64x256x56x56 {
  func.func @f_18_resnet_64x256x56x56_64x256x56x56_64x256x56x56(%input0: tensor<64x256x56x56xf32>, %input1: tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xf32> {
    %result = stablehlo.add %input0, %input1 : tensor<64x256x56x56xf32>
    return %result : tensor<64x256x56x56xf32>
  }
}
