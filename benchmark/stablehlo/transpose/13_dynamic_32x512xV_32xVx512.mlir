module @f_13_dynamic_32x512xV_32xVx512 {
  func.func @f_13_dynamic_32x512xV_32xVx512(%input: tensor<32x512x?xf32>) -> tensor<32x?x512xf32> {
    %result = stablehlo.transpose %input, dims = [0, 2, 1] : (tensor<32x512x?xf32>) -> tensor<32x?x512xf32>
    return %result : tensor<32x?x512xf32>
  }
}
