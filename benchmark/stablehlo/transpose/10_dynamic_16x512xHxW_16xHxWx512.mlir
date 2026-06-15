module @f_10_dynamic_16x512xHxW_16xHxWx512 {
  func.func @f_10_dynamic_16x512xHxW_16xHxWx512(%input: tensor<16x512x?x?xf32>) -> tensor<16x?x?x512xf32> {
    %result = stablehlo.transpose %input, dims = [0, 2, 3, 1] : (tensor<16x512x?x?xf32>) -> tensor<16x?x?x512xf32>
    return %result : tensor<16x?x?x512xf32>
  }
}
