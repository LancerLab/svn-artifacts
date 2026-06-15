module {
  func.func @f_12_dynamic_64xTx256_64Tx256(%input: tensor<64x?x256xf32>) -> tensor<?x256xf32> {
    %out = tensor.collapse_shape %input [[0, 1], [2]] : tensor<64x?x256xf32> into tensor<?x256xf32>
    return %out : tensor<?x256xf32>
  }
}
