module {
  func.func @f_5_dynamic_Nx256x56x56_Nx256x56x56_Nx256x56x56_correct(%a: tensor<?x256x56x56xf32>, %b: tensor<?x256x56x56xf32>) -> tensor<?x256x56x56xf32> {
    %c0 = arith.constant 0 : index
    %d0 = tensor.dim %a, %c0 : tensor<?x256x56x56xf32>
    %init = tensor.empty(%d0) : tensor<?x256x56x56xf32>
    %r = linalg.add ins(%a, %b : tensor<?x256x56x56xf32>, tensor<?x256x56x56xf32>)
                    outs(%init : tensor<?x256x56x56xf32>) -> tensor<?x256x56x56xf32>
    return %r : tensor<?x256x56x56xf32>
  }
}
