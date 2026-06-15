module @f_18_resnet_64x256x56x56_64x56x56x256 {
  func.func @f_18_resnet_64x256x56x56_64x56x56x256(%input: tensor<64x256x56x56xf32>) -> tensor<64x56x56x256xf32> {
    %result = stablehlo.transpose %input, dims = [0, 2, 3, 1] : (tensor<64x256x56x56xf32>) -> tensor<64x56x56x256xf32>
    return %result : tensor<64x56x56x256xf32>
  }
}
