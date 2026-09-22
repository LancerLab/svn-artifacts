module {
  func.func @f_ps(%plo: tensor<?x4xf32>, %phi: tensor<?x4xf32>, %core: tensor<4x4xf32>) -> tensor<20x4xf32> {
    %c0 = arith.constant 0 : index
    %cC = arith.constant 4 : index
    %c0f = arith.constant 0.0 : f32
    %low = tensor.dim %plo, %c0 : tensor<?x4xf32>
    %high = tensor.dim %phi, %c0 : tensor<?x4xf32>
    %init = tensor.empty() : tensor<20x4xf32>
    %z = linalg.fill ins(%c0f : f32) outs(%init : tensor<20x4xf32>) -> tensor<20x4xf32>
    %i1 = tensor.insert_slice %plo into %z[0, 0][%low, 4][1, 1] : tensor<?x4xf32> into tensor<20x4xf32>
    %coff = arith.addi %c0, %low : index
    %i2 = tensor.insert_slice %core into %i1[%coff, 0][4, 4][1, 1] : tensor<4x4xf32> into tensor<20x4xf32>
    %hoff = arith.addi %coff, %cC : index
    %i3 = tensor.insert_slice %phi into %i2[%hoff, 0][%high, 4][1, 1] : tensor<?x4xf32> into tensor<20x4xf32>
    return %i3 : tensor<20x4xf32>
  }
}
