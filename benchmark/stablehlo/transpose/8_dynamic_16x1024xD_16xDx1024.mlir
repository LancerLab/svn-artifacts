module @f_8_dynamic_16x1024xD_16xDx1024 {
  func.func @f_8_dynamic_16x1024xD_16xDx1024(%input: tensor<16x1024x?xf32>) -> tensor<16x?x1024xf32> {
    %result = stablehlo.transpose %input, dims = [0, 2, 1] : (tensor<16x1024x?xf32>) -> tensor<16x?x1024xf32>
    return %result : tensor<16x?x1024xf32>
  }
}
