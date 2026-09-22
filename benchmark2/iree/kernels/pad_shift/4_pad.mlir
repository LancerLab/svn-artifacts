module {
  func.func @f_ps(%plo: tensor<?x1xf32>, %phi: tensor<?x1xf32>, %core: tensor<8x1xf32>) -> tensor<24x1xf32> {
    %c0 = arith.constant 0 : index
    %cC = arith.constant 8 : index
    %c0f = arith.constant 0.0 : f32
    %low = tensor.dim %plo, %c0 : tensor<?x1xf32>
    %high = tensor.dim %phi, %c0 : tensor<?x1xf32>
    %init = tensor.empty() : tensor<24x1xf32>
    %z = linalg.fill ins(%c0f : f32) outs(%init : tensor<24x1xf32>) -> tensor<24x1xf32>
    %i1 = tensor.insert_slice %plo into %z[0, 0][%low, 1][1, 1] : tensor<?x1xf32> into tensor<24x1xf32>
    %coff = arith.addi %c0, %low : index
    %i2 = tensor.insert_slice %core into %i1[%coff, 0][8, 1][1, 1] : tensor<8x1xf32> into tensor<24x1xf32>
    %hoff = arith.addi %coff, %cC : index
    %i3 = tensor.insert_slice %phi into %i2[%hoff, 0][%high, 1][1, 1] : tensor<?x1xf32> into tensor<24x1xf32>
    return %i3 : tensor<24x1xf32>
  }
}
