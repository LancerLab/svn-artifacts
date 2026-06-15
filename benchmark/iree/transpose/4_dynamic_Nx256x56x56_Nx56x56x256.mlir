module {
  func.func @f_4_dynamic_Nx256x56x56_Nx56x56x256(%input: tensor<?x256x56x56xf32>) -> tensor<?x56x56x256xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<?x256x56x56xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<?x256x56x56xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<?x256x56x56xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<?x256x56x56xf32>
    %out = tensor.empty(%input_d0) : tensor<?x56x56x256xf32>
    %result = linalg.transpose ins(%input : tensor<?x256x56x56xf32>) outs(%out : tensor<?x56x56x256xf32>) permutation = [0, 2, 3, 1]
    return %result : tensor<?x56x56x256xf32>
  }
}
