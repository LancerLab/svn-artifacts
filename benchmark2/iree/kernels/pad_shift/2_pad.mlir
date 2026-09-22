module {
  func.func @f_ps(%plo: tensor<?x2xf32>, %phi: tensor<?x2xf32>, %core: tensor<2x2xf32>) -> tensor<18x2xf32> {
    %c0 = arith.constant 0 : index
    %cC = arith.constant 2 : index
    %c0f = arith.constant 0.0 : f32
    %low = tensor.dim %plo, %c0 : tensor<?x2xf32>
    %high = tensor.dim %phi, %c0 : tensor<?x2xf32>
    %init = tensor.empty() : tensor<18x2xf32>
    %z = linalg.fill ins(%c0f : f32) outs(%init : tensor<18x2xf32>) -> tensor<18x2xf32>
    %i1 = tensor.insert_slice %plo into %z[0, 0][%low, 2][1, 1] : tensor<?x2xf32> into tensor<18x2xf32>
    %coff = arith.addi %c0, %low : index
    %i2 = tensor.insert_slice %core into %i1[%coff, 0][2, 2][1, 1] : tensor<2x2xf32> into tensor<18x2xf32>
    %hoff = arith.addi %coff, %cC : index
    %i3 = tensor.insert_slice %phi into %i2[%hoff, 0][%high, 2][1, 1] : tensor<?x2xf32> into tensor<18x2xf32>
    return %i3 : tensor<18x2xf32>
  }
}
