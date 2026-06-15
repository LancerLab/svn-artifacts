module @f_9_dynamic_64x128xHxW_64xHxWx128 {
  func.func @f_9_dynamic_64x128xHxW_64xHxWx128(%input: tensor<64x128x?x?xf32>) -> tensor<64x?x?x128xf32> {
    %result = stablehlo.transpose %input, dims = [0, 2, 3, 1] : (tensor<64x128x?x?xf32>) -> tensor<64x?x?x128xf32>
    return %result : tensor<64x?x?x128xf32>
  }
}
