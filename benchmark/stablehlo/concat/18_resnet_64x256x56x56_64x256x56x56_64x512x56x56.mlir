module @f_18_resnet_64x256x56x56_64x256x56x56_64x512x56x56 {
  func.func @f_18_resnet_64x256x56x56_64x256x56x56_64x512x56x56(%in0: tensor<64x256x56x56xf32>, %in1: tensor<64x256x56x56xf32>) -> tensor<64x512x56x56xf32> {
    %result = stablehlo.concatenate %in0, %in1, dim = 1 : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x512x56x56xf32>
    return %result : tensor<64x512x56x56xf32>
  }
}
