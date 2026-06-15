module @f_5_dynamic_Nx1280xHxW_NxHxWx1280 {
  func.func @f_5_dynamic_Nx1280xHxW_NxHxWx1280(%input: tensor<?x1280x?x?xf32>) -> tensor<?x?x?x1280xf32> {
    %result = stablehlo.transpose %input, dims = [0, 2, 3, 1] : (tensor<?x1280x?x?xf32>) -> tensor<?x?x?x1280xf32>
    return %result : tensor<?x?x?x1280xf32>
  }
}
