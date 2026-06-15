module {
  func.func @f_4_dynamic_Nx256x56x56_Nx256x56x56_Nx512x56x56_correct(%in0: tensor<?x256x56x56xf32>, %in1: tensor<?x256x56x56xf32>) -> tensor<?x512x56x56xf32> {
    %r = tensor.concat dim(1) %in0, %in1 : (tensor<?x256x56x56xf32>, tensor<?x256x56x56xf32>) -> tensor<?x512x56x56xf32>
    return %r : tensor<?x512x56x56xf32>
  }
}
