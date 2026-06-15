module @f_4_dynamic_Nx256x56x56_Nx56x56x256 {
  func.func @f_4_dynamic_Nx256x56x56_Nx56x56x256(%input: tensor<?x256x56x56xf32>) -> tensor<?x56x56x256xf32> {
    %result = stablehlo.transpose %input, dims = [0, 2, 3, 1] : (tensor<?x256x56x56xf32>) -> tensor<?x56x56x256xf32>
    return %result : tensor<?x56x56x256xf32>
  }
}
