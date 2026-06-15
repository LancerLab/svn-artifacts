module @f_2_cnn_128x128x28x28_128x28x28x128 {
  func.func @f_2_cnn_128x128x28x28_128x28x28x128(%input: tensor<128x128x28x28xf32>) -> tensor<128x28x28x128xf32> {
    %result = stablehlo.transpose %input, dims = [0, 2, 3, 1] : (tensor<128x128x28x28xf32>) -> tensor<128x28x28x128xf32>
    return %result : tensor<128x28x28x128xf32>
  }
}
