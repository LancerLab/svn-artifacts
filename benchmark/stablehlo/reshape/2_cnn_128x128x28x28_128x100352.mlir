module @f_2_cnn_128x128x28x28_128x100352 {
  func.func @f_2_cnn_128x128x28x28_128x100352(%input: tensor<128x128x28x28xf32>) -> tensor<128x100352xf32> {
    %result = stablehlo.reshape %input : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    return %result : tensor<128x100352xf32>
  }
}
