module {
  func.func @f_4_dynamic_Nx256x56x56_Nx56x56x256_linalg(%input: tensor<?x256x56x56xf32>) -> tensor<?x56x56x256xf32> {
    %c0_0 = arith.constant 0 : index
    %d0 = tensor.dim %input, %c0_0 : tensor<?x256x56x56xf32>
    %init = tensor.empty(%d0) : tensor<?x56x56x256xf32>
    %r = linalg.transpose ins(%input : tensor<?x256x56x56xf32>) outs(%init : tensor<?x56x56x256xf32>) permutation = [0, 2, 3, 1]
    return %r : tensor<?x56x56x256xf32>
  }
}
