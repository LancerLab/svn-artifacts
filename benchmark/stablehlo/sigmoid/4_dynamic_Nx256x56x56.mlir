module @f_4_dynamic_Nx256x56x56 {
  func.func @f_4_dynamic_Nx256x56x56(%input: tensor<?x256x56x56xf32>) -> tensor<?x256x56x56xf32> {
    %result = stablehlo.logistic %input : tensor<?x256x56x56xf32>
    return %result : tensor<?x256x56x56xf32>
  }
}
