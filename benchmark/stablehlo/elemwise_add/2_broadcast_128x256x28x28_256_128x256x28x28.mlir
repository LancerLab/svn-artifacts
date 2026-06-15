module @f_2_broadcast_128x256x28x28_256_128x256x28x28 {
  func.func @f_2_broadcast_128x256x28x28_256_128x256x28x28(%input0: tensor<128x256x28x28xf32>, %input1: tensor<256xf32>) -> tensor<128x256x28x28xf32> {
    %b_bcast = stablehlo.broadcast_in_dim %input1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x28x28xf32>
    %result = stablehlo.add %input0, %b_bcast : tensor<128x256x28x28xf32>
    return %result : tensor<128x256x28x28xf32>
  }
}
