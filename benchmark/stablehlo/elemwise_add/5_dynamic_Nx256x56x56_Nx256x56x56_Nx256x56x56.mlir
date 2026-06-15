module @f_5_dynamic_Nx256x56x56_Nx256x56x56_Nx256x56x56 {
  func.func @f_5_dynamic_Nx256x56x56_Nx256x56x56_Nx256x56x56(%input0: tensor<?x256x56x56xf32>, %input1: tensor<?x256x56x56xf32>) -> tensor<?x256x56x56xf32> {
    %result = stablehlo.add %input0, %input1 : tensor<?x256x56x56xf32>
    return %result : tensor<?x256x56x56xf32>
  }
}
