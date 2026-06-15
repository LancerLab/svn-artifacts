module @f_2_cnn_128x128x28x28 {
  func.func @f_2_cnn_128x128x28x28(%input: tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32> {
    %result = stablehlo.logistic %input : tensor<128x128x28x28xf32>
    return %result : tensor<128x128x28x28xf32>
  }
}
