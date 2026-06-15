module @f_4_dynamic_Nx256x56x56_Nx256x56x56_Nx512x56x56 {
  func.func @f_4_dynamic_Nx256x56x56_Nx256x56x56_Nx512x56x56(%in0: tensor<?x256x56x56xf32>, %in1: tensor<?x256x56x56xf32>) -> tensor<?x512x56x56xf32> {
    %result = stablehlo.concatenate %in0, %in1, dim = 1 : (tensor<?x256x56x56xf32>, tensor<?x256x56x56xf32>) -> tensor<?x512x56x56xf32>
    return %result : tensor<?x512x56x56xf32>
  }
}
