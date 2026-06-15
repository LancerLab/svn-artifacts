module @f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28 {
  func.func @f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28(%in0: tensor<128x128x28x28xf32>, %in1: tensor<128x256x28x28xf32>) -> tensor<128x384x28x28xf32> {
    %result = stablehlo.concatenate %in0, %in1, dim = 1 : (tensor<128x128x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<128x384x28x28xf32>
    return %result : tensor<128x384x28x28xf32>
  }
}
