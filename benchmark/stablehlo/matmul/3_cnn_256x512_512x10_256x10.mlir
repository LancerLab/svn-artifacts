module @f_3_cnn_256x512_512x10_256x10 {
  func.func @f_3_cnn_256x512_512x10_256x10(%input0: tensor<256x512xf32>, %input1: tensor<512x10xf32>) -> tensor<256x10xf32> {
    %result = stablehlo.dot %input0, %input1 : (tensor<256x512xf32>, tensor<512x10xf32>) -> tensor<256x10xf32>
    return %result : tensor<256x10xf32>
  }
}
