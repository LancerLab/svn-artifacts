module {
  func.func @f_ps(%plo: tensor<?x8xf32>, %phi: tensor<?x8xf32>, %core: tensor<5x8xf32>) -> tensor<21x8xf32> {
    %c0 = arith.constant 0 : index
    %cC = arith.constant 5 : index
    %c0f = arith.constant 0.0 : f32
    %low = tensor.dim %plo, %c0 : tensor<?x8xf32>
    %high = tensor.dim %phi, %c0 : tensor<?x8xf32>
    %init = tensor.empty() : tensor<21x8xf32>
    %z = linalg.fill ins(%c0f : f32) outs(%init : tensor<21x8xf32>) -> tensor<21x8xf32>
    %i1 = tensor.insert_slice %plo into %z[0, 0][%low, 8][1, 1] : tensor<?x8xf32> into tensor<21x8xf32>
    %coff = arith.addi %c0, %low : index
    %i2 = tensor.insert_slice %core into %i1[%coff, 0][5, 8][1, 1] : tensor<5x8xf32> into tensor<21x8xf32>
    %hoff = arith.addi %coff, %cC : index
    %i3 = tensor.insert_slice %phi into %i2[%hoff, 0][%high, 8][1, 1] : tensor<?x8xf32> into tensor<21x8xf32>
    return %i3 : tensor<21x8xf32>
  }
}
