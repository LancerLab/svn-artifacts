module @f_5_dynamic_Nx1280xMxK {
  func.func @f_5_dynamic_Nx1280xMxK(%input: tensor<?x1280x?x?xf32>) -> tensor<?x1280x?x?xf32> {
    %result = stablehlo.logistic %input : tensor<?x1280x?x?xf32>
    return %result : tensor<?x1280x?x?xf32>
  }
}
