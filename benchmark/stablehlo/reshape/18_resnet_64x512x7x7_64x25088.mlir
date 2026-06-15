module @f_18_resnet_64x512x7x7_64x25088 {
  func.func @f_18_resnet_64x512x7x7_64x25088(%input: tensor<64x512x7x7xf32>) -> tensor<64x25088xf32> {
    %result = stablehlo.reshape %input : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    return %result : tensor<64x25088xf32>
  }
}
